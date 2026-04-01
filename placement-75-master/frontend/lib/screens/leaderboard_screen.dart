import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:ui';
import 'package:provider/provider.dart';
import 'package:placement_assistant/widgets/loading_overlay.dart';
import '../providers/auth_provider.dart';

class LeaderboardScreen extends StatefulWidget {
  const LeaderboardScreen({super.key});

  @override
  State<LeaderboardScreen> createState() => _LeaderboardScreenState();
}

class _LeaderboardScreenState extends State<LeaderboardScreen> {
  List<dynamic> leaders = [];
  bool isLoading = true;
  String? branch;

  @override
  void initState() {
    super.initState();
    _fetchLeaderboard();
  }

  Future<void> _fetchLeaderboard() async {
    final auth = Provider.of<AuthProvider>(context, listen: false);
    
    // First get user branch from dashboard if not stored in auth
    try {
      final dashRes = await http.get(Uri.parse('${auth.baseUrl}/dashboard/${auth.username}'));
      if (dashRes.statusCode == 200) {
        final dashData = json.decode(dashRes.body);
        branch = dashData['branch'];
      }

      final response = await http.get(Uri.parse('${auth.baseUrl}/leaderboard/$branch'));
      if (response.statusCode == 200) {
        setState(() {
          leaders = json.decode(response.body);
          isLoading = false;
        });
      }
    } catch (e) {
      print("Leaderboard Error: $e");
      setState(() => isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (isLoading) return PremiumLoadingOverlay(message: "Calculating Rankings...");

    return Scaffold(
      backgroundColor: const Color(0xFF0F172A),
      body: Stack(
        children: [
          // Background Glows
          Positioned(top: -100, right: -50, child: _blurGlow(Colors.indigo.withOpacity(0.2), 300)),
          Positioned(bottom: 100, left: -50, child: _blurGlow(Colors.blue.withOpacity(0.15), 250)),
          
          CustomScrollView(
            slivers: [
              _buildAppBar(),
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildHeader(),
                      const SizedBox(height: 24),
                      if (leaders.isEmpty)
                        _emptyState()
                      else
                        _buildLeaderList(),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildAppBar() {
    return const SliverAppBar(
      backgroundColor: Colors.transparent,
      elevation: 0,
      expandedHeight: 80,
      floating: true,
      centerTitle: false,
      title: Text("Wall of Fame", style: TextStyle(color: Colors.white, fontWeight: FontWeight.w800, fontSize: 24)),
      iconTheme: IconThemeData(color: Colors.white),
    );
  }

  Widget _buildHeader() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          "Top Performers in $branch",
          style: const TextStyle(color: Colors.white70, fontSize: 16),
        ),
        const SizedBox(height: 8),
        const Text(
          "Ranked by placement readiness and consistency",
          style: TextStyle(color: Colors.white38, fontSize: 13),
        ),
      ],
    );
  }

  Widget _buildLeaderList() {
    return ListView.separated(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: leaders.length,
      separatorBuilder: (context, index) => const SizedBox(height: 12),
      itemBuilder: (context, index) {
        final user = leaders[index];
        final bool isCurrentUser = user['username'] == Provider.of<AuthProvider>(context, listen: false).username;
        
        return _leaderCard(index + 1, user, isCurrentUser);
      },
    );
  }

  Widget _leaderCard(int rank, dynamic user, bool isMe) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: isMe ? Colors.blueAccent.withOpacity(0.15) : Colors.white.withOpacity(0.05),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: isMe ? Colors.blueAccent.withOpacity(0.5) : Colors.white.withOpacity(0.1),
              width: isMe ? 2 : 1,
            ),
          ),
          child: Row(
            children: [
              _rankBadge(rank),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      user['username'],
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                        decoration: isMe ? TextDecoration.underline : null,
                      ),
                    ),
                    Text(
                      "Lv. ${user['level']} Specialist",
                      style: const TextStyle(color: Colors.white38, fontSize: 12),
                    ),
                  ],
                ),
              ),
              _scoreMetric("Readiness", "${user['readiness_score']}%", Colors.greenAccent),
              const SizedBox(width: 12),
              _scoreMetric("Badges", "${user['badges_count']}", Colors.amberAccent),
            ],
          ),
        ),
      ),
    );
  }

  Widget _rankBadge(int rank) {
    Color color = Colors.white24;
    double size = 32;
    if (rank == 1) color = Colors.amber;
    if (rank == 2) color = Colors.grey[400]!;
    if (rank == 3) color = Colors.orange[800]!;

    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: color.withOpacity(0.2),
        shape: BoxShape.circle,
        border: Border.all(color: color.withOpacity(0.5), width: 2),
      ),
      child: Center(
        child: Text(
          "$rank",
          style: TextStyle(color: color, fontWeight: FontWeight.bold, fontSize: 14),
        ),
      ),
    );
  }

  Widget _scoreMetric(String label, String value, Color color) {
    return Column(
      children: [
        Text(value, style: TextStyle(color: color, fontWeight: FontWeight.bold, fontSize: 16)),
        Text(label, style: const TextStyle(color: Colors.white38, fontSize: 10)),
      ],
    );
  }

  Widget _emptyState() {
    return Center(
      child: Column(
        children: [
          const SizedBox(height: 40),
          Icon(Icons.emoji_events_outlined, color: Colors.white10, size: 80),
          const SizedBox(height: 16),
          const Text("No competition yet!", style: TextStyle(color: Colors.white38)),
        ],
      ),
    );
  }

  Widget _blurGlow(Color color, double size) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(
            color: color,
            blurRadius: 100,
            spreadRadius: 50,
          ),
        ],
      ),
    );
  }
}
