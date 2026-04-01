import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:http/http.dart' as http;
import 'package:fl_chart/fl_chart.dart';
import 'dart:convert';
import 'package:url_launcher/url_launcher.dart';
import 'package:placement_assistant/widgets/loading_overlay.dart';
import '../providers/auth_provider.dart';
import '../api_config.dart';
import 'quiz_screen.dart';
import 'interview_screen.dart'; 
import 'gd_screen.dart'; // Fixed import name to match standard file naming
import 'performance_graph.dart';
import 'leaderboard_screen.dart';
import 'dart:async';
import 'package:flutter_tts/flutter_tts.dart';
import '../widgets/branch_selection_dialog.dart';
import '../widgets/news_notification.dart';
import 'login_screen.dart';
import 'daily_report_screen.dart';
import 'instructions_screen.dart';
import '../widgets/teacher_message_notification.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});
  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final ScrollController _dashboardScrollController = ScrollController();
  Map<String, dynamic>? data;
  Map<String, dynamic>? weeklyData;
  List<dynamic>? newsData;
  List<dynamic>? _suggestions;
  bool loading = true;
  bool newsLoading = false;
  bool briefingLoading = false;
  bool isPlayingBriefing = false;
  int _selectedIndex = 0; // Tracks Sidebar selection
  bool _trendsShown = false; // Flag to show trends only once per session
  Timer? _newsTimer;
  int _currentNewsIndex = 0;
  final FlutterTts _flutterTts = FlutterTts();
  OverlayEntry? _newsOverlayEntry;
  OverlayEntry? _suggestionOverlayEntry;

  @override
  void initState() {
    super.initState();
    loadData();
    // Use a delay to show trends popup after the UI settles
    Future.delayed(const Duration(milliseconds: 800), () {
      if (mounted) _showTrendsPopup();
    });
    _loadNews().then((_) => _startNewsTimer());
  }

  @override
  void dispose() {
    _newsTimer?.cancel();
    _dashboardScrollController.dispose();
    _newsOverlayEntry?.remove();
    _newsOverlayEntry = null;
    _suggestionOverlayEntry?.remove();
    _suggestionOverlayEntry = null;
    super.dispose();
  }

  void _startNewsTimer() {
    _newsTimer = Timer.periodic(const Duration(minutes: 5), (timer) {
      if (newsData != null && newsData!.isNotEmpty) {
        _showNewsNotification(newsData![_currentNewsIndex]);
        _currentNewsIndex = (_currentNewsIndex + 1) % newsData!.length;
      }
    });
  }

  void _showNewsNotification(dynamic item) {
    if (_newsOverlayEntry != null) {
      _newsOverlayEntry?.remove();
      _newsOverlayEntry = null;
    }

    _newsOverlayEntry = OverlayEntry(
      builder: (context) => Positioned(
        top: 50,
        right: 0,
        child: NewsNotification(
          item: item,
          onDismiss: () {
            _newsOverlayEntry?.remove();
            _newsOverlayEntry = null;
          },
          onRead: (text) async {
            await _flutterTts.setLanguage("en-US");
            await _flutterTts.setPitch(1.0);
            await _flutterTts.speak(text);
          },
        ),
      ),
    );

    Overlay.of(context).insert(_newsOverlayEntry!);

    // Automatically dismiss after 15 seconds
    Future.delayed(const Duration(seconds: 15), () {
      if (_newsOverlayEntry != null) {
        _newsOverlayEntry?.remove();
        _newsOverlayEntry = null;
      }
    });
  }

  Future<void> loadData() async {
    final auth = Provider.of<AuthProvider>(context, listen: false);
    try {
      final res1 = await http.get(Uri.parse('${auth.baseUrl}/dashboard/${auth.username}'));
      final res2 = await http.get(Uri.parse('${auth.baseUrl}/weekly_report/${auth.username}'));

      if (res1.statusCode == 200) {
        setState(() {
          data = jsonDecode(res1.body);
          if (res2.statusCode == 200) weeklyData = jsonDecode(res2.body);
        });
      }
      if (auth.username != null) {
        final suggestionsList = await ApiConfig.fetchSuggestions(auth.username!);
        setState(() {
          _suggestions = suggestionsList;
        });
        _showUnreadSuggestions(suggestionsList);
      }
    } catch (e) {
      debugPrint("LOAD DATA ERROR: $e");
    } finally {
      if (mounted) setState(() => loading = false);
    }
  }

  Future<void> _loadNews() async {
    if (newsData != null) return; // Only load once or if forced
    setState(() => newsLoading = true);
    try {
      final news = await ApiConfig.fetchLatestNews();
      setState(() {
        newsData = news;
        newsLoading = false;
      });
    } catch (e) {
      debugPrint("NEWS ERROR: $e");
      setState(() => newsLoading = false);
    }
  }

  void _showUnreadSuggestions(List<dynamic> suggestions) {
    final unread = suggestions.where((s) => s['is_read'] == false).toList();
    if (unread.isNotEmpty) {
      // Show the latest unread suggestion
      _showSuggestionNotification(unread.first);
    }
  }

  void _showSuggestionNotification(dynamic suggestion) {
    if (_suggestionOverlayEntry != null) {
      _suggestionOverlayEntry?.remove();
      _suggestionOverlayEntry = null;
    }

    _suggestionOverlayEntry = OverlayEntry(
      builder: (context) => Positioned(
        top: 50,
        left: 20, // Different position from news to avoid overlap
        child: TeacherMessageNotification(
          suggestion: suggestion,
          onDismiss: () {
            _suggestionOverlayEntry?.remove();
            _suggestionOverlayEntry = null;
          },
        ),
      ),
    );

    Overlay.of(context).insert(_suggestionOverlayEntry!);
  }

  Future<void> _playIndustryBriefing() async {
    if (isPlayingBriefing) {
      await _flutterTts.stop();
      setState(() => isPlayingBriefing = false);
      return;
    }

    setState(() => briefingLoading = true);
    try {
      final briefing = await ApiConfig.fetchIndustryTrendsBriefing();
      setState(() {
        briefingLoading = false;
        isPlayingBriefing = true;
      });
      await _flutterTts.setLanguage("en-US");
      await _flutterTts.speak(briefing);

      _flutterTts.setCompletionHandler(() {
        if (mounted) {
          setState(() => isPlayingBriefing = false);
        }
      });
    } catch (e) {
      debugPrint("Briefing ERROR: $e");
      setState(() {
        briefingLoading = false;
        isPlayingBriefing = false;
      });
    }
  }

  Widget _buildDashboardSection(AuthProvider auth) {
    if (data == null) return const Center(child: CircularProgressIndicator());

    return Scrollbar(
      controller: _dashboardScrollController,
      thumbVisibility: true,
      trackVisibility: true,
      child: SingleChildScrollView(
        controller: _dashboardScrollController,
        padding: const EdgeInsets.all(32),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildTopSection(auth),
            const SizedBox(height: 32),
            _buildMiddleSection(context),
            const SizedBox(height: 32),
            const SizedBox(height: 32),
            _buildFocusAreas(),
            const SizedBox(height: 32),
            _buildBottomSection(),
          ],
        ),
      ),
    );
  }

  Widget _buildMessagesSection() {
    if (_suggestions == null || _suggestions!.isEmpty) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(height: 16),
            const Text("No messages from teachers yet", style: TextStyle(color: Colors.white24, fontSize: 16)),
          ],
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text("Messages from Teachers", style: TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold)),
          const Text("Direct feedback and suggestions to improve your performance", style: TextStyle(color: Colors.white54, fontSize: 14)),
          const SizedBox(height: 24),
          Expanded(
            child: ListView.builder(
              primary: true,
              itemCount: _suggestions!.length,
              itemBuilder: (context, index) {
                final s = _suggestions![index];
                final bool isRead = s['is_read'] == 1 || s['is_read'] == true;
                
                return Container(
                  margin: const EdgeInsets.only(bottom: 16),
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: const Color(0xFF161625),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: isRead ? Colors.white.withOpacity(0.05) : Colors.tealAccent.withOpacity(0.3),
                      width: isRead ? 1 : 1.5,
                    ),
                    boxShadow: isRead ? [] : [
                      BoxShadow(color: Colors.tealAccent.withOpacity(0.05), blurRadius: 10, offset: const Offset(0, 4))
                    ],
                  ),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Container(
                        child: const SizedBox.shrink(),
                      ),
                      const SizedBox(width: 20),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  "Teacher: ${s['teacher'] ?? 'Academic Staff'}",
                                  style: TextStyle(
                                    fontWeight: FontWeight.bold,
                                    color: isRead ? Colors.white70 : Colors.tealAccent,
                                    fontSize: 16,
                                  ),
                                ),
                                Text(
                                  s['timestamp'] != null ? s['timestamp'].toString().substring(0, 10) : "",
                                  style: const TextStyle(color: Colors.white38, fontSize: 11),
                                ),
                              ],
                            ),
                            const SizedBox(height: 12),
                            Text(
                              s['message'] ?? "",
                              style: const TextStyle(color: Colors.white, fontSize: 15, height: 1.5),
                            ),
                            const SizedBox(height: 16),
                            if (!isRead)
                              Align(
                                alignment: Alignment.centerRight,
                                child: TextButton(
                                  onPressed: () async {
                                    await ApiConfig.markSuggestionRead(s['id']);
                                    loadData(); // Refresh list
                                  },
                                   child: const Text("Mark as Read"),
                                  style: TextButton.styleFrom(
                                    foregroundColor: Colors.tealAccent,
                                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                                  ),
                                ),
                              ),
                          ],
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  // --- TOP SECTION: Level & Weekly Progress ---
  Widget _buildTopSection(AuthProvider auth) {
    double progress = (data?['weekly_progress'] ?? 0.0) / 100.0;
    final int streak = weeklyData?['streak'] ?? 0;
    final int branchRank = weeklyData?['branch_rank'] ?? 0;
    
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: const LinearGradient(colors: [Color(0xFF2196F3), Color(0xFF1E88E5)]),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [BoxShadow(color: Colors.blue.withOpacity(0.3), blurRadius: 20, offset: const Offset(0, 10))],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text("Current Level", style: TextStyle(color: Colors.white70, fontSize: 14)),
                  Text("Level ${data?['technical_level'] ?? 1}", style: const TextStyle(color: Colors.white, fontSize: 28, fontWeight: FontWeight.bold)),
                ],
              ),
              if (branchRank > 0)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.2), 
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: Colors.white.withOpacity(0.4)),
                  ),
                  child: Text(
                    "Rank: #$branchRank in ${data?['branch'] ?? 'Branch'}", 
                    style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)
                  ),
                )
              else
                const Icon(Icons.stars, color: Colors.white, size: 40),
            ],
          ),
          if (streak > 0) ...[
            const SizedBox(height: 15),
            Row(
              children: [
                const Text("🔥", style: TextStyle(fontSize: 20)),
                const SizedBox(width: 8),
                Text("$streak Day Practice Streak", style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)),
              ],
            ),
          ],
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text("Weekly Progress", style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
              Text("${(progress * 100).toInt()}%", style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(10),
            child: LinearProgressIndicator(
              value: progress,
              backgroundColor: Colors.white24,
              color: Colors.white,
              minHeight: 12,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAchievementsSection() {
    final badges = weeklyData?['badges'] ?? [];
    if (badges.isEmpty) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text("Your Achievements", style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold)),
        const SizedBox(height: 16),
        SizedBox(
          height: 100,
          child: ListView.separated(
            scrollDirection: Axis.horizontal,
            itemCount: badges.length,
            separatorBuilder: (context, index) => const SizedBox(width: 16),
            itemBuilder: (context, index) {
              final badge = badges[index];
              return _badgeItem(badge['name'], badge['icon'], _getBadgeColor(badge['color']));
            },
          ),
        ),
      ],
    );
  }

  Color _getBadgeColor(String colorName) {
    switch (colorName) {
      case 'gold': return Colors.amber;
      case 'orange': return Colors.orangeAccent;
      case 'blue': return Colors.blueAccent;
      case 'purple': return Colors.blueAccent;
      default: return Colors.grey;
    }
  }

  Widget _badgeItem(String label, String iconName, Color color) {
    IconData iconData;
    switch (iconName) {
      case 'emoji_events': iconData = Icons.emoji_events; break;
      case 'whatshot': iconData = Icons.whatshot; break;
      case 'record_voice_over': iconData = Icons.record_voice_over; break;
      case 'face': iconData = Icons.face; break;
      default: iconData = Icons.stars;
    }

    return Container(
      width: 100,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF161625),
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(iconData, color: color, size: 30),
          const SizedBox(height: 8),
          Text(
            label,
            textAlign: TextAlign.center,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  // --- MIDDLE SECTION: Today's Tasks ---
  Widget _buildMiddleSection(BuildContext context) {
    final tasks = data?['tasks'] ?? {};
    bool aptDone = tasks['aptitude_done'] ?? false;
    bool techDone = tasks['tech_done'] ?? false;
    bool interviewDone = tasks['interview_done'] ?? false;
    bool gdDone = tasks['gd_done'] ?? false;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text("Today's Practice", style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold)),
        const SizedBox(height: 16),
        Row(
          children: [
            _taskItem("Aptitude (10 questions)", aptDone, Colors.orangeAccent, () => _startQuiz(context, "APTITUDE")),
            const SizedBox(width: 20),
            _taskItem("Technical (10 questions)", techDone, Colors.blueAccent, () => _startQuiz(context, "TECHNICAL")),
          ],
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            _taskItem("Mock Interview", interviewDone, Colors.purpleAccent, () {
              Navigator.push(context, MaterialPageRoute(
                builder: (context) => InstructionsScreen(
                  category: "INTERVIEW",
                  onStart: () {
                    Navigator.pushReplacement(context, MaterialPageRoute(builder: (context) => const InterviewScreen()));
                  },
                ),
              ));
            }),
            const SizedBox(width: 20),
            _taskItem("Group Discussion", gdDone, Colors.cyanAccent, () {
              Navigator.push(context, MaterialPageRoute(
                builder: (context) => InstructionsScreen(
                  category: "GD",
                  onStart: () {
                    Navigator.pushReplacement(context, MaterialPageRoute(builder: (context) => const GdScreen()));
                  },
                ),
              ));
            }),
          ],
        ),
      ],
    );
  }

  Widget _taskItem(String label, bool isDone, Color color, VoidCallback onTap) {
    return Expanded(
      child: InkWell(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: const Color(0xFF161625),
            borderRadius: BorderRadius.circular(15),
            border: Border.all(color: isDone ? color : Colors.white10),
          ),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(color: color.withOpacity(0.1), shape: BoxShape.circle),
                child: Icon(isDone ? Icons.check : Icons.play_arrow, color: color),
              ),
              const SizedBox(width: 15),
              Expanded(child: Text(label, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold))),
              if (isDone) const Icon(Icons.verified, color: Colors.greenAccent, size: 20),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFocusAreas() {
    List techWeak = data?['weak_areas_tech'] ?? [];
    List aptWeak = data?['weak_areas_apt'] ?? [];
    String techStatus = data?['weak_areas_tech_status'] ?? "active";
    String aptStatus = data?['weak_areas_apt_status'] ?? "active";
    
    // Status overrules list content for UI messaging
    bool isCollecting = (techStatus == "collecting" || aptStatus == "collecting");

    int techLvl = data?['technical_level'] ?? 1;
    int aptLvl = data?['aptitude_level'] ?? 1;

    String getLvlName(int l) {
      if (l == 1) return "Easy";
      if (l == 2) return "Medium";
      if (l == 3) return "Hard";
      return "Company-level";
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text("Focus Areas", style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold)),
        const SizedBox(height: 16),
        if (isCollecting)
           Container(
             width: double.infinity,
             padding: const EdgeInsets.all(20),
             decoration: BoxDecoration(color: const Color(0xFF161625), borderRadius: BorderRadius.circular(15), border: Border.all(color: Colors.white10)),
             child: const Row(
               children: [
                 Icon(Icons.insights, color: Colors.blueAccent),
                 SizedBox(width: 15),
                 Expanded(
                   child: Text(
                     "Collecting daily performance data to identify your focus areas. Keep practicing!",
                     style: TextStyle(color: Colors.white70, fontSize: 13, fontStyle: FontStyle.italic),
                   ),
                 ),
               ],
             ),
           )
        else
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _focusCard("Technical", techWeak, Colors.blueAccent, getLvlName(techLvl)),
              const SizedBox(width: 20),
              _focusCard("Aptitude", aptWeak, Colors.orangeAccent, getLvlName(aptLvl)),
            ],
          ),
      ],
    );
  }

  Widget _focusCard(String title, List areas, Color color, String level) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(color: const Color(0xFF161625), borderRadius: BorderRadius.circular(15)),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(title, style: TextStyle(color: color, fontWeight: FontWeight.bold, fontSize: 16)),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(color: color.withOpacity(0.1), borderRadius: BorderRadius.circular(8)),
                  child: Text(level, style: TextStyle(color: color, fontSize: 10, fontWeight: FontWeight.bold)),
                ),
              ],
            ),
            const SizedBox(height: 15),
            if (areas.isEmpty)
              const Text("Great job! No weak areas found.", style: TextStyle(color: Colors.white38, fontSize: 12, fontStyle: FontStyle.italic))
            else
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: areas.map((a) {
                  final areaName = a is Map ? (a['area'] ?? "None") : a.toString();
                  return Container(
                    margin: const EdgeInsets.only(bottom: 8),
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.05),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.white10)
                    ),
                    child: Text(
                      areaName,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 13,
                        fontWeight: FontWeight.w500
                      ),
                    ),
                  );
                }).toList(),
              ),
          ],
        ),
      ),
    );
  }

  Widget _countTag(String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color.withOpacity(0.2)),
      ),
      child: Text(
        text,
        style: TextStyle(color: color, fontSize: 9, fontWeight: FontWeight.bold),
      ),
    );
  }

  // --- BOTTOM SECTION: Graphs & Reports ---
  Widget _buildBottomSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text("Performance Report", style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold)),
            TextButton(
              onPressed: () {
                Navigator.push(context, MaterialPageRoute(builder: (context) => PerformanceGraph()));
              },
              child: const Text("View Full Analytics", style: TextStyle(color: Colors.blueAccent)),
            )
          ],
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            _miniStat("Total Attempted", "${data?['total_attempts'] ?? 0}", Icons.assignment_outlined, Colors.cyanAccent),
            _miniStat("Avg Accuracy", "${data?['accuracy'] ?? 0}%", Icons.track_changes, Colors.greenAccent),
          ],
        ),
        const SizedBox(height: 24),
        _buildGraphSection(),
      ],
    );
  }

  Widget _miniStat(String label, String value, IconData icon, Color color) {
    return Expanded(
      child: Container(
        margin: const EdgeInsets.only(right: 15),
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(color: const Color(0xFF161625), borderRadius: BorderRadius.circular(15)),
        child: Row(
          children: [
            Icon(icon, color: color, size: 24),
            const SizedBox(width: 15),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(value, style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
                Text(label, style: const TextStyle(color: Colors.white38, fontSize: 12)),
              ],
            )
          ],
        ),
      ),
    );
  }

  Widget _buildNewsSection() {
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("Industry Trends", style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white)),
                  Text("Latest updates from Hacker News", style: TextStyle(color: Colors.white54)),
                ],
              ),
              IconButton(
                icon: const Icon(Icons.refresh, color: Colors.blueAccent),
                onPressed: () {
                  setState(() => newsData = null);
                  _loadNews();
                },
              ),
            ],
          ),
          const SizedBox(height: 16),
          _buildBriefingBanner(),
          const SizedBox(height: 20),
          Expanded(
            child: newsLoading
                ? const Center(child: CircularProgressIndicator(color: Colors.blueAccent))
                : newsData == null || newsData!.isEmpty
                    ? const Center(child: Text("No trends found", style: TextStyle(color: Colors.white38)))
                    : ListView.builder(
                        itemCount: newsData!.length,
                        itemBuilder: (context, index) {
                          final item = newsData![index];
                          return _newsCardItem(item);
                        },
                      ),
          ),
        ],
      ),
    );
  }

  Widget _buildBriefingBanner() {
    return InkWell(
      onTap: briefingLoading ? null : _playIndustryBriefing,
      borderRadius: BorderRadius.circular(15),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.blueAccent.withOpacity(0.2), Color(0xFF1E88E5).withOpacity(0.2)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(15),
          border: Border.all(color: Colors.blueAccent.withOpacity(0.5)),
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: Colors.blueAccent.withOpacity(0.2),
                shape: BoxShape.circle,
              ),
              child: briefingLoading 
                ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(color: Colors.blueAccent, strokeWidth: 2))
                : Icon(
                    isPlayingBriefing ? Icons.stop_circle_outlined : Icons.multitrack_audio, 
                    color: Colors.blueAccent, 
                    size: 24
                  ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    isPlayingBriefing ? "Playing Briefing..." : "Listen to Industry AI Briefing",
                    style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  const Text(
                    "A personalized 3-sentence summary of current trends.",
                    style: TextStyle(color: Colors.white54, fontSize: 12),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _newsCardItem(dynamic item, {bool isPopup = false}) {
    return Card(
      color: isPopup ? Colors.white.withOpacity(0.08) : Colors.white.withOpacity(0.05),
      margin: const EdgeInsets.only(bottom: 12),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      child: ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        title: Text(item['title'] ?? 'No Title', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: isPopup ? 14 : 16)),
        subtitle: Padding(
          padding: const EdgeInsets.only(top: 4.0),
          child: Text(
            "Points: ${item['score']} • By: ${item['by']}",
            style: const TextStyle(color: Colors.white38, fontSize: 11),
          ),
        ),
        onTap: () => showNewsSummaryDialog(context, item),
      ),
    );
  }

  OverlayEntry? _trendsOverlayEntry;

  void _showTrendsPopup() async {
    if (_trendsShown) return;
    _trendsShown = true;

    // Fetch in background, then show a side panel
    final data = await ApiConfig.fetchLatestNews().then((v) => v.take(4).toList()).catchError((_) => <dynamic>[]);
    if (!mounted) return;

    _trendsOverlayEntry = OverlayEntry(
      builder: (context) => Positioned(
        top: 80,
        right: 16,
        child: _TrendsSidePanel(
          items: data,
          onDismiss: () {
            _trendsOverlayEntry?.remove();
            _trendsOverlayEntry = null;
          },
        ),
      ),
    );
    Overlay.of(context).insert(_trendsOverlayEntry!);

    // Auto-dismiss after 15 seconds
    Future.delayed(const Duration(seconds: 15), () {
      if (_trendsOverlayEntry?.mounted ?? false) {
        _trendsOverlayEntry?.remove();
        _trendsOverlayEntry = null;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    if (loading) return PremiumLoadingOverlay(message: "Initializing Dashboard...");
    
    final auth = Provider.of<AuthProvider>(context);
    const Color sidebarBg = Color(0xFF161625);
    const Color scaffoldBg = Color(0xFF0F0C29);

    return Scaffold(
      backgroundColor: scaffoldBg,
      body: Row(
        children: [
          // --- SIDEBAR (Left Panel) ---
          Container(
            width: 280,
            color: sidebarBg,
            child: Column(
              children: [
                const SizedBox(height: 40),
                _buildSidebarHeader(),
                const SizedBox(height: 20),
                
                _buildProfileCard(auth.username ?? "User"),
                
                const SizedBox(height: 30),
                _sidebarTile(0, Icons.dashboard_outlined, "Dashboard"),
                _sidebarTile(1, Icons.code, "Technical Practice", onTap: () => _startQuiz(context, "TECHNICAL")),
                _sidebarTile(2, Icons.psychology_outlined, "Aptitude Practice", onTap: () => _startQuiz(context, "APTITUDE")),
                
                _sidebarTile(3, Icons.groups_outlined, "GD Practice", onTap: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => InstructionsScreen(
                    category: 'GD',
                    onStart: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const GdScreen())),
                  )));
                }),
                
                _sidebarTile(4, Icons.mic_none, "Interview Practice", onTap: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => InstructionsScreen(
                    category: 'INTERVIEW',
                    onStart: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const InterviewScreen())),
                  )));
                }),

                _sidebarTile(5, Icons.auto_graph, "Industry Trends", onTap: () {
                  setState(() => _selectedIndex = 5);
                  _loadNews();
                }),
                
                _sidebarTile(6, Icons.calendar_month, "Report", onTap: () {
                  setState(() => _selectedIndex = 6);
                }),

                _sidebarTile(7, Icons.message_outlined, "Messages", onTap: () {
                  setState(() => _selectedIndex = 7);
                  loadData(); // Refresh to get latest messages
                }),

                const Spacer(),
                _sidebarTile(-1, Icons.logout, "Logout", onTap: () {
                  final auth = Provider.of<AuthProvider>(context, listen: false);
                  auth.logout();
                  Navigator.pushAndRemoveUntil(
                    context,
                    MaterialPageRoute(builder: (context) => const LoginScreen()),
                    (Route<dynamic> route) => false,
                  );
                }),
                const SizedBox(height: 20),
              ],
            ),
          ),

          // --- MAIN CONTENT AREA ---
          Expanded(
            child: _selectedIndex == 5 
                ? _buildNewsSection()
                : _selectedIndex == 6
                    ? const DailyReportScreen()
                    : _selectedIndex == 7
                        ? _buildMessagesSection()
                        : _buildDashboardSection(auth),
          ),
        ],
      ),
    );
  }

  // --- UI COMPONENTS ---

  Widget _buildSidebarHeader() {
    return const Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.auto_awesome, color: Colors.blueAccent),
        SizedBox(width: 10),
        Text("AI Placement", style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 18)),
      ],
    );
  }

  Widget _buildProfileCard(String name) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20),
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(color: Colors.white.withOpacity(0.05), borderRadius: BorderRadius.circular(15)),
      child: Row(
        children: [
          const CircleAvatar(backgroundColor: Colors.blueAccent, child: Icon(Icons.person, color: Colors.white)),
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(name, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
              Text("${data?['branch'] ?? 'Set Branch'} • Student", style: const TextStyle(color: Colors.white38, fontSize: 11)),
            ],
          )
        ],
      ),
    );
  }

  Widget _sidebarTile(int index, IconData icon, String label, {VoidCallback? onTap, bool isComingSoon = false}) {
    bool isSelected = _selectedIndex == index;
    return ListTile(
      onTap: isComingSoon ? null : (onTap ?? () => setState(() => _selectedIndex = index)),
      leading: Icon(icon, color: isSelected ? Colors.blueAccent : Colors.white54),
      title: Text(label, style: TextStyle(color: isSelected ? Colors.white : Colors.white54, fontSize: 14)),
      trailing: isComingSoon ? Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
        decoration: BoxDecoration(color: Colors.white10, borderRadius: BorderRadius.circular(5)),
        child: const Text("Soon", style: TextStyle(color: Colors.white38, fontSize: 10)),
      ) : null,
    );
  }

  Widget _statCard(String label, String value, IconData icon, Color color) {
    return Expanded(
      child: Container(
        margin: const EdgeInsets.only(right: 15),
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(color: const Color(0xFF161625), borderRadius: BorderRadius.circular(20)),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: color, size: 28),
            const SizedBox(height: 15),
            Text(value, style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white)),
            Text(label, style: const TextStyle(color: Colors.white38, fontSize: 12)),
          ],
        ),
      ),
    );
  }

  Widget _buildGraphSection() {
    Map<int, FlSpot> aptitudeSpotsMap = {};
    Map<int, FlSpot> technicalSpotsMap = {};
    Map<int, FlSpot> interviewSpotsMap = {};
    Map<int, FlSpot> gdSpotsMap = {};
    final List<String> days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    
    if (data != null && data!['current_week_daily'] != null) {
      final dailyData = data!['current_week_daily'] as Map;
      final int todayWeekday = DateTime.now().weekday - 1; // 0=Mon, 6=Sun
      
      if (dailyData['aptitude'] != null) {
        final list = dailyData['aptitude'] as List;
        for (var item in list) {
          DateTime dt = DateTime.parse(item['day']);
          int weekday = dt.weekday - 1;
          if (weekday <= todayWeekday) {
            aptitudeSpotsMap[weekday] = FlSpot(weekday.toDouble(), double.tryParse(item['score'].toString()) ?? 0);
          }
        }
      }
      
      if (dailyData['technical'] != null) {
        final list = dailyData['technical'] as List;
        for (var item in list) {
          DateTime dt = DateTime.parse(item['day']);
          int weekday = dt.weekday - 1;
          if (weekday <= todayWeekday) {
            technicalSpotsMap[weekday] = FlSpot(weekday.toDouble(), double.tryParse(item['score'].toString()) ?? 0);
          }
        }
      }

      if (dailyData['interview'] != null) {
        final list = dailyData['interview'] as List;
        for (var item in list) {
          DateTime dt = DateTime.parse(item['day']);
          int weekday = dt.weekday - 1;
          if (weekday <= todayWeekday) {
            interviewSpotsMap[weekday] = FlSpot(weekday.toDouble(), double.tryParse(item['score'].toString()) ?? 0);
          }
        }
      }

      if (dailyData['gd'] != null) {
        final list = dailyData['gd'] as List;
        for (var item in list) {
          DateTime dt = DateTime.parse(item['day']);
          int weekday = dt.weekday - 1;
          if (weekday <= todayWeekday) {
            gdSpotsMap[weekday] = FlSpot(weekday.toDouble(), double.tryParse(item['score'].toString()) ?? 0);
          }
        }
      }
    }

    List<FlSpot> aptitudeSpots = aptitudeSpotsMap.values.toList()..sort((a, b) => a.x.compareTo(b.x));
    List<FlSpot> technicalSpots = technicalSpotsMap.values.toList()..sort((a, b) => a.x.compareTo(b.x));
    List<FlSpot> interviewSpots = interviewSpotsMap.values.toList()..sort((a, b) => a.x.compareTo(b.x));
    List<FlSpot> gdSpots = gdSpotsMap.values.toList()..sort((a, b) => a.x.compareTo(b.x));

    return Container(
      height: 300,
      padding: const EdgeInsets.all(25),
      decoration: BoxDecoration(color: const Color(0xFF161625), borderRadius: BorderRadius.circular(20)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text("Recent Performance (First Attempts)", style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
              Row(
                children: [
                   _legendItem("Aptitude", Colors.cyanAccent),
                   const SizedBox(width: 12),
                   _legendItem("Technical", Colors.orangeAccent),
                   const SizedBox(width: 12),
                   _legendItem("Interview", Colors.pinkAccent),
                   const SizedBox(width: 12),
                   _legendItem("GD", Colors.greenAccent),
                ],
              )
            ],
          ),
          const SizedBox(height: 20),
          Expanded(
            child: LineChart(
              LineChartData(
                minY: 0,
                maxY: 10,
                minX: 0,
                maxX: 6,
                gridData: FlGridData(
                  show: true,
                  drawVerticalLine: false,
                  getDrawingHorizontalLine: (value) => const FlLine(color: Colors.white10, strokeWidth: 1),
                ),
                titlesData: FlTitlesData(
                  topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      interval: 1,
                      getTitlesWidget: (value, meta) {
                        int index = value.toInt();
                        if (index >= 0 && index < days.length) {
                          return Text(days[index], style: const TextStyle(color: Colors.white38, fontSize: 10));
                        }
                        return const Text("");
                      },
                    ),
                  ),
                ),
                borderData: FlBorderData(show: false),
                lineBarsData: [
                  if (aptitudeSpots.isNotEmpty)
                    LineChartBarData(
                      spots: aptitudeSpots,
                      isCurved: aptitudeSpots.length > 1,
                      color: Colors.cyanAccent,
                      barWidth: 3,
                      dotData: FlDotData(
                        show: true,
                        getDotPainter: (spot, percent, barData, index) => FlDotCirclePainter(
                          radius: 4,
                          color: Colors.cyanAccent,
                          strokeWidth: 2,
                          strokeColor: const Color(0xFF161625),
                        ),
                      ),
                      belowBarData: BarAreaData(show: true, color: Colors.cyanAccent.withOpacity(0.05)),
                    ),
                  if (technicalSpots.isNotEmpty)
                    LineChartBarData(
                      spots: technicalSpots,
                      isCurved: technicalSpots.length > 1,
                      color: Colors.orangeAccent,
                      barWidth: 3,
                      dotData: FlDotData(
                        show: true,
                        getDotPainter: (spot, percent, barData, index) => FlDotCirclePainter(
                          radius: 4,
                          color: Colors.orangeAccent,
                          strokeWidth: 2,
                          strokeColor: const Color(0xFF161625),
                        ),
                      ),
                      belowBarData: BarAreaData(show: true, color: Colors.orangeAccent.withOpacity(0.05)),
                    ),
                  if (interviewSpots.isNotEmpty)
                    LineChartBarData(
                      spots: interviewSpots,
                      isCurved: interviewSpots.length > 1,
                      color: Colors.pinkAccent,
                      barWidth: 3,
                      dotData: FlDotData(
                        show: true,
                        getDotPainter: (spot, percent, barData, index) => FlDotCirclePainter(
                          radius: 4,
                          color: Colors.pinkAccent,
                          strokeWidth: 2,
                          strokeColor: const Color(0xFF161625),
                        ),
                      ),
                      belowBarData: BarAreaData(show: true, color: Colors.pinkAccent.withOpacity(0.05)),
                    ),
                  if (gdSpots.isNotEmpty)
                    LineChartBarData(
                      spots: gdSpots,
                      isCurved: gdSpots.length > 1,
                      color: Colors.greenAccent,
                      barWidth: 3,
                      dotData: FlDotData(
                        show: true,
                        getDotPainter: (spot, percent, barData, index) => FlDotCirclePainter(
                          radius: 4,
                          color: Colors.greenAccent,
                          strokeWidth: 2,
                          strokeColor: const Color(0xFF161625),
                        ),
                      ),
                      belowBarData: BarAreaData(show: true, color: Colors.greenAccent.withOpacity(0.05)),
                    ),
                  // Fallback empty line to keep axes if both empty
                  if (aptitudeSpots.isEmpty && technicalSpots.isEmpty && interviewSpots.isEmpty && gdSpots.isEmpty)
                    LineChartBarData(
                      spots: [const FlSpot(0, 0), const FlSpot(6, 0)],
                      color: Colors.transparent,
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _legendItem(String label, Color color) {
    return Row(
      children: [
        Container(width: 12, height: 12, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
        const SizedBox(width: 5),
        Text(label, style: const TextStyle(color: Colors.white70, fontSize: 11)),
      ],
    );
  }



  void _startQuiz(BuildContext context, String cat) async {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => InstructionsScreen(
          category: cat,
          onStart: () async {
            if (cat == "TECHNICAL") {
              // For Technical quiz, show branch selection first
              if (!context.mounted) return;
              final selectedBranch = await showDialog<String>(
                context: context,
                barrierDismissible: false,
                builder: (_) => BranchSelectionDialog(initialBranch: data?['branch'], practiceMode: true),
              );
              if (selectedBranch != null && context.mounted) {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => QuizScreen(category: cat, targetBranch: selectedBranch)),
                ).then((_) => loadData());
              }
            } else {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => QuizScreen(category: cat)),
              ).then((_) => loadData());
            }
          },
        ),
      ),
    );
  }

  void _showBranchSelection() {
    showDialog(
      context: context,
      barrierDismissible: false, // Force them to choose
      builder: (_) => BranchSelectionDialog(initialBranch: data?['branch']),
    ).then((selected) {
      if (selected != null) {
        loadData(); // Reload to get updated branch
      }
    });
  }
}

// ────────────────────────────────────────────────────────────────────────────
// SESSION HISTORY
// ────────────────────────────────────────────────────────────────────────────

class _SessionHistoryView extends StatefulWidget {
  final String username;
  const _SessionHistoryView({required this.username});

  @override
  State<_SessionHistoryView> createState() => _SessionHistoryViewState();
}

class _SessionHistoryViewState extends State<_SessionHistoryView> {
  List<dynamic>? _sessions;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    final auth = Provider.of<AuthProvider>(context, listen: false);
    try {
      final res = await http.get(Uri.parse('${auth.baseUrl}/results/${widget.username}'));
      if (res.statusCode == 200) {
        setState(() {
          _sessions = jsonDecode(res.body);
          _loading = false;
        });
      } else {
        setState(() => _loading = false);
      }
    } catch (e) {
      setState(() => _loading = false);
    }
  }

  String _getCategoryIcon(String cat) {
    switch (cat.toUpperCase()) {
      case 'INTERVIEW': return '🎤';
      case 'APTITUDE': return '🧠';
      case 'TECHNICAL': return '💻';
      case 'GD': return '💬';
      default: return '📝';
    }
  }

  Color _getCategoryColor(String cat) {
    switch (cat.toUpperCase()) {
      case 'INTERVIEW': return Colors.blueAccent;
      case 'APTITUDE': return Colors.cyanAccent;
      case 'TECHNICAL': return Colors.orangeAccent;
      case 'GD': return Colors.greenAccent;
      default: return Colors.white54;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.history_edu_outlined, color: Colors.blueAccent, size: 28),
              const SizedBox(width: 12),
              const Text("Session History", style: TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold)),
              const Spacer(),
              IconButton(
                icon: const Icon(Icons.refresh, color: Colors.white38),
                onPressed: () { setState(() => _loading = true); _loadHistory(); },
              ),
            ],
          ),
          const SizedBox(height: 8),
          const Text("Tap any session to view your full report.", style: TextStyle(color: Colors.white38, fontSize: 13)),
          const SizedBox(height: 24),
          if (_loading)
            const Center(child: CircularProgressIndicator(color: Colors.blueAccent))
          else if (_sessions == null || _sessions!.isEmpty)
            Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const SizedBox(height: 80),
                  const Icon(Icons.inbox_outlined, color: Colors.white24, size: 64),
                  const SizedBox(height: 16),
                  const Text("No sessions yet.", style: TextStyle(color: Colors.white38, fontSize: 16)),
                  const SizedBox(height: 8),
                  const Text("Complete a quiz or interview to see your history here.", style: TextStyle(color: Colors.white24, fontSize: 13)),
                ],
              ),
            )
          else
            Expanded(
              child: RefreshIndicator(
                onRefresh: _loadHistory,
                child: ListView.builder(
                  primary: true,
                  itemCount: _sessions!.length,
                  itemBuilder: (context, index) {
                    final s = _sessions![index];
                    final cat = s['category'] as String? ?? 'N/A';
                    final score = double.tryParse(s['score'].toString()) ?? 0.0;
                    final color = _getCategoryColor(cat);

                    return Container(
                      margin: const EdgeInsets.only(bottom: 12),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.03),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: color.withOpacity(0.15)),
                      ),
                      child: ListTile(
                        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                        leading: Container(
                          width: 50,
                          height: 50,
                          decoration: BoxDecoration(
                            color: color.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(14),
                          ),
                          child: Center(child: Text(_getCategoryIcon(cat), style: const TextStyle(fontSize: 22))),
                        ),
                        title: Text(
                          "${cat.toUpperCase()}  —  ${s['area'] ?? 'General'}",
                          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600, fontSize: 14),
                        ),
                        subtitle: Padding(
                          padding: const EdgeInsets.only(top: 4),
                          child: Text(
                            "${s['date']?.toString().substring(0, 16) ?? ''}".replaceAll('T', ' '),
                            style: const TextStyle(color: Colors.white38, fontSize: 11),
                          ),
                        ),
                        trailing: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                          decoration: BoxDecoration(
                            color: color.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            "${score.toStringAsFixed(1)} / 10",
                            style: TextStyle(color: color, fontWeight: FontWeight.bold, fontSize: 13),
                          ),
                        ),
                        onTap: () => _showReportDialog(context, s),
                      ),
                    );
                  },
                ),
              ),
            ),
        ],
      ),
    );
  }

  void _showReportDialog(BuildContext context, Map<String, dynamic> session) {
    final auth = Provider.of<AuthProvider>(context, listen: false);
    showDialog(
      context: context,
      builder: (_) => _ReportDetailDialog(
        session: session,
        baseUrl: auth.baseUrl,
      ),
    );
  }
}

// ─── REPORT DETAIL DIALOG ───────────────────────────────────────────────────

class _ReportDetailDialog extends StatefulWidget {
  final Map<String, dynamic> session;
  final String baseUrl;
  const _ReportDetailDialog({required this.session, required this.baseUrl});

  @override
  State<_ReportDetailDialog> createState() => _ReportDetailDialogState();
}

class _ReportDetailDialogState extends State<_ReportDetailDialog> {
  Map<String, dynamic>? _detail;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _fetchDetail();
  }

  Future<void> _fetchDetail() async {
    try {
      final id = widget.session['id'];
      final cat = widget.session['category'];
      final res = await http.get(Uri.parse('${widget.baseUrl}/session_detail/$cat/$id'));
      if (res.statusCode == 200) {
        setState(() { _detail = jsonDecode(res.body); _loading = false; });
      } else {
        setState(() => _loading = false);
      }
    } catch (e) {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final cat = (widget.session['category'] as String? ?? '').toUpperCase();
    return Dialog(
      backgroundColor: const Color(0xFF161625),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 780, maxHeight: 720),
        child: Padding(
          padding: const EdgeInsets.all(28),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header
              Row(
                children: [
                  Expanded(
                    child: Text(
                      "$cat Session Report",
                      style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    "${(double.tryParse(widget.session['score'].toString()) ?? 0.0).toStringAsFixed(1)} / 10",
                    style: const TextStyle(color: Colors.blueAccent, fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(width: 12),
                  IconButton(
                    icon: const Icon(Icons.close, color: Colors.white54),
                    onPressed: () => Navigator.pop(context),
                  ),
                ],
              ),
              Text(
                "Area: ${widget.session['area'] ?? 'General'}  •  ${(widget.session['date'] as String? ?? '').replaceAll('T', ' ').substring(0, 16)}",
                style: const TextStyle(color: Colors.white38, fontSize: 12),
              ),
              const SizedBox(height: 16),
              const Divider(color: Colors.white10),
              const SizedBox(height: 12),
              if (_loading)
                const Expanded(child: Center(child: CircularProgressIndicator(color: Colors.blueAccent)))
              else if (_detail == null)
                const Expanded(child: Center(child: Text("Could not load report.", style: TextStyle(color: Colors.white38))))
              else
                Expanded(child: SingleChildScrollView(child: _buildDetailBody(cat))),
              const SizedBox(height: 16),
              Align(
                alignment: Alignment.centerRight,
                child: TextButton.icon(
                  onPressed: () => Navigator.pop(context),
                  icon: const Icon(Icons.check, color: Colors.blueAccent),
                  label: const Text("Close", style: TextStyle(color: Colors.blueAccent)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDetailBody(String cat) {
    final d = _detail!;

    if (cat == 'INTERVIEW') {
      final behavioral = d['behavioral_report'] as Map<String, dynamic>? ?? {};
      final scores = d['scores'] as Map<String, dynamic>? ?? {};
      final techReport = (d['technical_report'] as List?) ?? [];

      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Score Cards
          Row(
            children: [
              _scoreChip("Technical", "${scores['tech'] ?? 'N/A'}", Colors.cyanAccent),
              const SizedBox(width: 10),
              _scoreChip("Voice", "${scores['voice'] ?? 'N/A'}", Colors.blueAccent),
              const SizedBox(width: 10),
              _scoreChip("Camera", "${scores['camera'] ?? 'N/A'}", Colors.blueAccent),
            ],
          ),
          if (behavioral['behavioral_feedback'] != null || behavioral['overall_confidence'] != null) ...[
            const SizedBox(height: 20),
            _sectionBox("Behavioral Summary", behavioral['behavioral_feedback']?.toString() ?? behavioral['overall_confidence']?.toString() ?? "N/A", Colors.blueAccent),
          ],
          const SizedBox(height: 20),
          const Text("QUESTION BREAKDOWN", style: TextStyle(color: Colors.white38, fontSize: 11, fontWeight: FontWeight.w900, letterSpacing: 1.5)),
          const SizedBox(height: 12),
          ...techReport.map((q) => Container(
            margin: const EdgeInsets.only(bottom: 14),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(color: Colors.white.withOpacity(0.03), borderRadius: BorderRadius.circular(16)),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(color: Colors.cyanAccent.withOpacity(0.1), borderRadius: BorderRadius.circular(6)),
                    child: Text(q['accuracy'] ?? "N/A", style: const TextStyle(color: Colors.cyanAccent, fontSize: 10, fontWeight: FontWeight.bold)),
                  ),
                  const SizedBox(width: 10),
                  Expanded(child: Text(q['question'] ?? "", style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600, fontSize: 14))),
                ]),
                const SizedBox(height: 12),
                _answerRow("YOUR RESPONSE", q['your_answer'] ?? "N/A", Colors.white70),
                const SizedBox(height: 10),
                _answerRow("IDEAL ANSWER", q['ideal_answer'] ?? "N/A", Colors.greenAccent),
                const SizedBox(height: 10),
                _answerRow("IMPROVEMENT", q['improvement'] ?? "N/A", Colors.orangeAccent),
              ],
            ),
          )),
        ],
      );
    } else if (cat == 'GD') {
      final scores = d['scores'] as Map<String, dynamic>? ?? {};
      return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          _scoreChip("Content", "${scores['content'] ?? 'N/A'}", Colors.greenAccent),
          const SizedBox(width: 10),
          _scoreChip("Communication", "${scores['communication'] ?? 'N/A'}", Colors.blueAccent),
          const SizedBox(width: 10),
          _scoreChip("Camera", "${scores['camera'] ?? 'N/A'}", Colors.blueAccent),
        ]),
        const SizedBox(height: 20),
        if (d['feedback'] != null) _sectionBox("Feedback", d['feedback'].toString(), Colors.greenAccent),
        if (d['ideal_answer'] != null) ...[
          const SizedBox(height: 16),
          _sectionBox("Ideal Response", d['ideal_answer'].toString(), Colors.cyanAccent),
        ],
        if (d['transcript'] != null) ...[
          const SizedBox(height: 16),
          _sectionBox("Your Transcript", d['transcript'].toString(), Colors.white70),
        ],
      ]);
    } else {
      // QUIZ (APTITUDE / TECHNICAL)
      final questions = (d['questions'] as List?) ?? [];
      int correct = questions.where((q) => q['is_correct'] == true).length;
      return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        _scoreChip("Score", "$correct / ${questions.length}", Colors.cyanAccent),
        const SizedBox(height: 20),
        const Text("QUESTION BREAKDOWN", style: TextStyle(color: Colors.white38, fontSize: 11, fontWeight: FontWeight.w900, letterSpacing: 1.5)),
        const SizedBox(height: 12),
        ...questions.asMap().entries.map((entry) {
          final i = entry.key;
          final q = entry.value;
          final isCorrect = q['is_correct'] == true;
          final badgeColor = isCorrect ? Colors.greenAccent : Colors.redAccent;
          return Container(
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: badgeColor.withOpacity(0.04),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: badgeColor.withOpacity(0.12)),
            ),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Row(children: [
                Icon(isCorrect ? Icons.check_circle_outline : Icons.cancel_outlined, color: badgeColor, size: 18),
                const SizedBox(width: 8),
                Expanded(child: Text("Q${i+1}. ${q['question'] ?? ''}", style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600, fontSize: 13))),
              ]),
              const SizedBox(height: 10),
              _answerRow("YOUR ANSWER", q['user_selected']?.toString() ?? "N/A", isCorrect ? Colors.greenAccent : Colors.redAccent),
              if (!isCorrect) ...[
                const SizedBox(height: 6),
                _answerRow("CORRECT ANSWER", q['correct_answer']?.toString() ?? "N/A", Colors.greenAccent),
              ],
              if (q['explanation'] != null) ...[
                const SizedBox(height: 6),
                _answerRow("EXPLANATION", q['explanation'].toString(), Colors.white38),
              ],
            ]),
          );
        }),
      ]);
    }
  }

  Widget _scoreChip(String label, String value, Color color) {
    return Flexible(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(color: color.withOpacity(0.08), borderRadius: BorderRadius.circular(12)),
        child: Column(children: [
          Text(value, style: TextStyle(color: color, fontSize: 18, fontWeight: FontWeight.bold)),
          Text(label, style: const TextStyle(color: Colors.white38, fontSize: 10, fontWeight: FontWeight.w700)),
        ]),
      ),
    );
  }

  Widget _sectionBox(String title, String content, Color color) {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(title, style: TextStyle(color: color.withOpacity(0.5), fontSize: 9, fontWeight: FontWeight.w900, letterSpacing: 1.2)),
      const SizedBox(height: 6),
      Container(
        width: double.infinity,
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(color: color.withOpacity(0.04), borderRadius: BorderRadius.circular(12)),
        child: Text(content, style: TextStyle(color: color, fontSize: 13, height: 1.5)),
      ),
    ]);
  }

  Widget _answerRow(String label, String text, Color color) {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(label, style: TextStyle(color: color.withOpacity(0.5), fontSize: 9, fontWeight: FontWeight.w900, letterSpacing: 1.2)),
      const SizedBox(height: 4),
      Text(text, style: TextStyle(color: color, fontSize: 13, height: 1.4)),
    ]);
  }
}

class _TrendsSidePanel extends StatefulWidget {
  final List<dynamic> items;
  final VoidCallback onDismiss;

  const _TrendsSidePanel({required this.items, required this.onDismiss});

  @override
  State<_TrendsSidePanel> createState() => _TrendsSidePanelState();
}

class _TrendsSidePanelState extends State<_TrendsSidePanel>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<Offset> _slide;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 350));
    _slide = Tween<Offset>(begin: const Offset(1, 0), end: Offset.zero)
        .animate(CurvedAnimation(parent: _ctrl, curve: Curves.easeOutCubic));
    _ctrl.forward();
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  void _dismiss() {
    _ctrl.reverse().then((_) => widget.onDismiss());
  }

  @override
  Widget build(BuildContext context) {
    return SlideTransition(
      position: _slide,
      child: Material(
        color: Colors.transparent,
        child: Container(
          width: 300,
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: const Color(0xFF1E1B3A),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: Colors.blueAccent.withOpacity(0.4)),
            boxShadow: [BoxShadow(color: Colors.blue.withOpacity(0.3), blurRadius: 20, offset: const Offset(-4, 4))],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                children: [
                  const Icon(Icons.trending_up, color: Colors.blueAccent, size: 18),
                  const SizedBox(width: 8),
                  const Expanded(
                    child: Text("Industry Trends",
                        style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 14)),
                  ),
                  GestureDetector(
                    onTap: _dismiss,
                    child: const Icon(Icons.close, color: Colors.white38, size: 16),
                  ),
                ],
              ),
              const SizedBox(height: 4),
              const Text("Top headlines right now", style: TextStyle(color: Colors.white38, fontSize: 10)),
              const Divider(height: 16, color: Colors.white12),
              if (widget.items.isEmpty)
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 8),
                  child: Text("No trends available", style: TextStyle(color: Colors.white38, fontSize: 12)),
                )
              else
                ...widget.items.map((item) => Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: InkWell(
                    onTap: () {
                      _dismiss();
                      showNewsSummaryDialog(context, item);
                    },
                    borderRadius: BorderRadius.circular(8),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Container(width: 4, height: 4, margin: const EdgeInsets.fromLTRB(0, 6, 8, 0),
                            decoration: const BoxDecoration(color: Colors.blueAccent, shape: BoxShape.circle)),
                        Expanded(child: Text(item['title'] ?? '', maxLines: 2, overflow: TextOverflow.ellipsis,
                            style: const TextStyle(color: Colors.white70, fontSize: 11, height: 1.4))),
                      ],
                    ),
                  ),
                )),
            ],
          ),
        ),
      ),
    );
  }
}

void showNewsSummaryDialog(BuildContext context, dynamic item) {
  showDialog(
    context: context,
    barrierDismissible: true,
    builder: (context) => FutureBuilder<String>(
      future: ApiConfig.fetchNewsSummary(item['title'] ?? '', url: item['url']),
      builder: (context, snapshot) {
        return AlertDialog(
          backgroundColor: const Color(0xFF161625),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          title: Text(item['title'] ?? 'News Summary', style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
          content: snapshot.connectionState == ConnectionState.waiting
              ? const SizedBox(height: 100, child: Center(child: CircularProgressIndicator(color: Colors.blueAccent)))
              : Text(
                  snapshot.data ?? "Failed to load summary.",
                  style: const TextStyle(color: Colors.white70, height: 1.5, fontSize: 14),
                ),
          actions: [
            TextButton(
              onPressed: () {
                ApiConfig.stopSpeech();
                Navigator.pop(context);
              },
              child: const Text("Close", style: TextStyle(color: Colors.blueAccent)),
            ),
            if (item['url'] != null)
              TextButton(
                onPressed: () async {
                  final url = Uri.parse(item['url']);
                  if (await canLaunchUrl(url)) {
                    await launchUrl(url, mode: LaunchMode.externalApplication);
                  }
                },
                child: const Text("Read Source", style: TextStyle(color: Colors.white38, fontSize: 12)),
              ),
          ],
        );
      },
    ),
  );
}
