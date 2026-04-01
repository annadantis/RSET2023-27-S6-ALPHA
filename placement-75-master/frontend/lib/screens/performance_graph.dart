import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:ui';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';
import 'package:placement_assistant/widgets/loading_overlay.dart';
import '../api_config.dart';
import '../widgets/report_detail_dialog.dart';

class PerformanceGraph extends StatefulWidget {
  @override
  _PerformanceGraphState createState() => _PerformanceGraphState();
}

class _PerformanceGraphState extends State<PerformanceGraph> {
  Map<String, dynamic> reportData = {};
  Map<String, dynamic> dashboardData = {};
  List<dynamic> history = [];
  bool isLoading = true;
  bool showLastWeek = false;
  DateTime? selectedDate;
  Map<String, dynamic>? customWeekData;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    final auth = Provider.of<AuthProvider>(context, listen: false);
    final String username = auth.username ?? "Student";

    try {
      final reportRes = await http.get(Uri.parse('${auth.baseUrl}/weekly_report/$username'));
      final dashRes = await http.get(Uri.parse('${auth.baseUrl}/dashboard/$username'));

      if (reportRes.statusCode == 200 && dashRes.statusCode == 200) {
        final historyData = await ApiConfig.fetchUserHistory(username);
        setState(() {
          reportData = json.decode(reportRes.body);
          dashboardData = json.decode(dashRes.body);
          history = historyData;
          selectedDate = null; // Reset on refresh
          customWeekData = null;
          isLoading = false;
        });
      }
    } catch (e) {
      print("Error fetching graph data: $e");
      setState(() => isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (isLoading) return PremiumLoadingOverlay(message: "Deep-Diving Into Your Analytics...");

    return Scaffold(
      backgroundColor: const Color(0xFF0F172A),
      body: Stack(
        children: [
          // Background Glows
          Positioned(top: -100, right: -50, child: _blurGlow(Colors.indigo.withOpacity(0.2), 300)),
          Positioned(bottom: 100, left: -50, child: _blurGlow(Colors.blue.withOpacity(0.15), 250)),
          
          CustomScrollView(
            primary: true,
            slivers: [
              _buildAppBar(),
              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                sliver: SliverList(
                  delegate: SliverChildListDelegate([
                    _buildReadinessHero(),
                    const SizedBox(height: 24),
                    _buildQuickStats(),
                    const SizedBox(height: 32),
                    _sectionHeader("Daily Momentum", "Performance trends for current week (First Attempts)"),
                    const SizedBox(height: 16),
                    _buildDailyChart(),
                    const SizedBox(height: 32),
                    _sectionHeader("Weekly Milestone Progress", "Holistic growth (Apt + Tech + GD + Interview)"),
                    const SizedBox(height: 16),
                    _buildCumulativeWeeklyChart(),
                    const SizedBox(height: 32),
                    _sectionHeader("Skill Profile", "Areas where you excel and areas for growth"),
                    const SizedBox(height: 16),
                    _buildSkillProfile(),
                    const SizedBox(height: 32),
                    const SizedBox(height: 32),
                    _sectionHeader("Session History", "Click on any session to view detailed AI feedback"),
                    const SizedBox(height: 16),
                    _buildRecentSessions(),
                    const SizedBox(height: 40),
                  ]),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildAppBar() {
    return SliverAppBar(
      backgroundColor: Colors.transparent,
      elevation: 0,
      expandedHeight: 80,
      floating: true,
      centerTitle: false,
      title: const Text("Full Analytics", style: TextStyle(color: Colors.white, fontWeight: FontWeight.w800, fontSize: 24)),
      iconTheme: const IconThemeData(color: Colors.white),
    );
  }

  Widget _sectionHeader(String title, String sub) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
        const SizedBox(height: 4),
        Text(sub, style: const TextStyle(color: Colors.white38, fontSize: 12)),
      ],
    );
  }

  Widget _buildReadinessHero() {
    double readiness = (reportData['readiness_score'] ?? 0.0).toDouble();
    String status = reportData['status'] ?? "Preparing";

    return _glassCard(
      padding: const EdgeInsets.all(24),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text("Placement Readiness", style: TextStyle(color: Colors.white60, fontSize: 14)),
                const SizedBox(height: 8),
                Text(status, style: const TextStyle(color: Colors.white, fontSize: 32, fontWeight: FontWeight.w900)),
                const SizedBox(height: 12),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(color: Colors.indigoAccent.withOpacity(0.2), borderRadius: BorderRadius.circular(20)),
                  child: const Text("Keep grinding to level up!", style: TextStyle(color: Colors.indigoAccent, fontSize: 11, fontWeight: FontWeight.bold)),
                ),
              ],
            ),
          ),
          Stack(
            alignment: Alignment.center,
            children: [
              SizedBox(
                width: 90,
                height: 90,
                child: CircularProgressIndicator(
                  value: readiness / 100,
                  strokeWidth: 10,
                  backgroundColor: Colors.white.withOpacity(0.1),
                  color: readiness > 75 ? Colors.greenAccent : (readiness > 40 ? Colors.indigoAccent : Colors.orangeAccent),
                ),
              ),
              Text("${readiness.toInt()}%", style: const TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.w900)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildQuickStats() {
    return Row(
      children: [
        _statItem("Attempts", "${dashboardData['total_attempts'] ?? 0}", Icons.auto_graph, Colors.cyanAccent),
        const SizedBox(width: 12),
        _statItem("Accuracy", "${dashboardData['accuracy'] ?? 0}%", Icons.track_changes, Colors.greenAccent),
        const SizedBox(width: 12),
        _statItem("Streak", "${reportData['streak'] ?? 0}d", Icons.whatshot, Colors.orangeAccent),
      ],
    );
  }

  Widget _statItem(String label, String value, IconData icon, Color color) {
    return Expanded(
      child: _glassCard(
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 12),
        child: Column(
          children: [
            Icon(icon, color: color.withOpacity(0.8), size: 18),
            const SizedBox(height: 8),
            Text(value, style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 2),
            Text(label, style: const TextStyle(color: Colors.white38, fontSize: 10)),
          ],
        ),
      ),
    );
  }

  Widget _buildDailyChart() {
    Map<int, FlSpot> aptitudeSpotsMap = {};
    Map<int, FlSpot> technicalSpotsMap = {};
    Map<int, FlSpot> interviewSpotsMap = {};
    Map<int, FlSpot> gdSpotsMap = {};
    
    final List<String> days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    
    // Logic for date range
    int maxWeekdayToShow = 6; // Default to full week
    if (selectedDate != null) {
      maxWeekdayToShow = selectedDate!.weekday - 1;
    } else if (!showLastWeek) {
      maxWeekdayToShow = DateTime.now().weekday - 1;
    }

    // Use custom data if date is selected, otherwise fallback to dashboard data
    final Map? dataToUse = selectedDate != null ? customWeekData : (dashboardData[showLastWeek ? 'last_week_daily' : 'current_week_daily'] as Map?);
    
    if (dataToUse != null) {
      void populateSpots(String key, Map<int, FlSpot> map) {
        if (dataToUse[key] != null) {
          final list = dataToUse[key] as List;
          for (var item in list) {
            DateTime dt = DateTime.parse(item['day']);
            int weekday = dt.weekday - 1;
            if (weekday <= maxWeekdayToShow) {
              map[weekday] = FlSpot(weekday.toDouble(), (double.tryParse(item['score'].toString()) ?? 0));
            }
          }
        }
      }

      populateSpots('aptitude', aptitudeSpotsMap);
      populateSpots('technical', technicalSpotsMap);
      populateSpots('interview', interviewSpotsMap);
      populateSpots('gd', gdSpotsMap);
    }

    List<FlSpot> aptitudeSpots = aptitudeSpotsMap.values.toList()..sort((a, b) => a.x.compareTo(b.x));
    List<FlSpot> technicalSpots = technicalSpotsMap.values.toList()..sort((a, b) => a.x.compareTo(b.x));
    List<FlSpot> interviewSpots = interviewSpotsMap.values.toList()..sort((a, b) => a.x.compareTo(b.x));
    List<FlSpot> gdSpots = gdSpotsMap.values.toList()..sort((a, b) => a.x.compareTo(b.x));

    return _glassCard(
      padding: const EdgeInsets.fromLTRB(10, 24, 20, 16),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Wrap(
                spacing: 12,
                children: [
                  _legendItem("Aptitude", Colors.cyanAccent),
                  _legendItem("Technical", Colors.orangeAccent),
                  _legendItem("Interview", Colors.pinkAccent),
                  _legendItem("GD", Colors.greenAccent),
                ],
              ),
              Row(
                children: [
                  InkWell(
                    onTap: () => setState(() {
                      showLastWeek = !showLastWeek;
                      selectedDate = null;
                    }),
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: showLastWeek ? Colors.indigoAccent : Colors.white.withOpacity(0.05),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: Colors.white.withOpacity(0.1))
                      ),
                      child: Text(showLastWeek ? "Prev Week" : "Curr Week", style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold)),
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton(
                    icon: Icon(Icons.calendar_month, color: selectedDate != null ? Colors.indigoAccent : Colors.white38),
                    onPressed: _pickDate,
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 24),
          SizedBox(
            height: 220,
            child: LineChart(
              LineChartData(
                minY: 0,
                maxY: 10,
                minX: 0,
                maxX: maxWeekdayToShow.toDouble(),
                gridData: FlGridData(
                  show: true,
                  drawVerticalLine: true,
                  getDrawingHorizontalLine: (v) => FlLine(color: Colors.white.withOpacity(0.05), strokeWidth: 1),
                  getDrawingVerticalLine: (v) => FlLine(color: Colors.white.withOpacity(0.05), strokeWidth: 1),
                ),
                titlesData: FlTitlesData(
                  leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, reservedSize: 30, getTitlesWidget: (v, m) => Text(v.toInt().toString(), style: const TextStyle(color: Colors.white24, fontSize: 10)))),
                  bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, interval: 1, getTitlesWidget: (v, m) {
                    int idx = v.toInt();
                    if (idx < 0 || idx >= days.length) return const SizedBox();
                    return Text(days[idx], style: const TextStyle(color: Colors.white24, fontSize: 10));
                  })),
                  rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                ),
                borderData: FlBorderData(show: false),
                lineBarsData: [
                  if (aptitudeSpots.isNotEmpty) _lineData(aptitudeSpots, Colors.cyanAccent),
                  if (technicalSpots.isNotEmpty) _lineData(technicalSpots, Colors.orangeAccent),
                  if (interviewSpots.isNotEmpty) _lineData(interviewSpots, Colors.pinkAccent),
                  if (gdSpots.isNotEmpty) _lineData(gdSpots, Colors.greenAccent),
                  if (aptitudeSpots.isEmpty && technicalSpots.isEmpty && interviewSpots.isEmpty && gdSpots.isEmpty)
                    LineChartBarData(
                      spots: [const FlSpot(0, 0), FlSpot(maxWeekdayToShow.toDouble(), 0)],
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

  Widget _buildCumulativeWeeklyChart() {
    List data = reportData['cumulative_weekly'] ?? [];
    List<FlSpot> spots = [];
    for (int i = 0; i < data.length; i++) {
      spots.add(FlSpot(i.toDouble(), (data[i]['score'] as num).toDouble()));
    }

    return _glassCard(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          const Text("Cumulative Growth (Apt + Tech + GD + Interview)", style: TextStyle(color: Colors.white54, fontSize: 11)),
          const SizedBox(height: 20),
          SizedBox(
            height: 180,
            child: LineChart(
              LineChartData(
                minY: 0,
                minX: 0,
                maxX: spots.isEmpty ? 3 : (spots.length - 1).toDouble(),
                gridData: FlGridData(show: true, drawVerticalLine: false, getDrawingHorizontalLine: (v) => FlLine(color: Colors.white.withOpacity(0.05), strokeWidth: 1)),
                titlesData: FlTitlesData(
                  leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, reservedSize: 30, getTitlesWidget: (v, m) => Text(v.toInt().toString(), style: const TextStyle(color: Colors.white24, fontSize: 10)))),
                  bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, interval: 1, getTitlesWidget: (v, m) => Text("W${v.toInt() + 1}", style: const TextStyle(color: Colors.white24, fontSize: 10)))),
                  rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                ),
                borderData: FlBorderData(show: false),
                lineBarsData: [
                  LineChartBarData(
                    spots: spots.isEmpty ? [const FlSpot(0, 0)] : spots,
                    isCurved: spots.length > 1,
                    color: Colors.indigoAccent,
                    barWidth: 4,
                    dotData: FlDotData(
                      show: true,
                      getDotPainter: (spot, percent, barData, index) => FlDotCirclePainter(
                        radius: 5,
                        color: Colors.indigoAccent,
                        strokeWidth: 2,
                        strokeColor: const Color(0xFF0F172A),
                      ),
                    ),
                    belowBarData: BarAreaData(show: true, gradient: LinearGradient(begin: Alignment.topCenter, end: Alignment.bottomCenter, colors: [Colors.indigoAccent.withOpacity(0.2), Colors.indigoAccent.withOpacity(0.01)])),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSkillProfile() {
    final strong = reportData['strong_areas'] ?? [];
    final needsFocus = (dashboardData['weak_areas_tech'] ?? []) + (dashboardData['weak_areas_apt'] ?? []);

    return Column(
      children: [
        _skillBlock("Top Strengths", strong, Colors.greenAccent),
        const SizedBox(height: 16),
        _skillBlock("Growth Areas", needsFocus, Colors.orangeAccent),
      ],
    );
  }

  Widget _skillBlock(String title, List items, Color color) {
    return _glassCard(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(width: 4, height: 16, decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(2))),
              const SizedBox(width: 8),
              Text(title, style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 16),
          if (items.isEmpty)
             const Text("Keep practicing to unlock insights...", style: TextStyle(color: Colors.white24, fontSize: 12, fontStyle: FontStyle.italic))
          else
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: items.map((i) {
                String label = i is Map ? i['area'].toString() : i.toString();
                return Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(color: color.withOpacity(0.1), borderRadius: BorderRadius.circular(8), border: Border.all(color: color.withOpacity(0.2))),
                  child: Text(label, style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w600)),
                );
              }).toList(),
            ),
        ],
      ),
    );
  }

  // --- Helpers ---
  Widget _glassCard({required Widget child, EdgeInsetsGeometry? padding}) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(24),
      child: Container(
        padding: padding,
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.03),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(color: Colors.white.withOpacity(0.08)),
        ),
        child: child,
      ),
    );
  }

  Widget _blurGlow(Color color, double size) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(color: color, shape: BoxShape.circle),
      child: BackdropFilter(filter: ImageFilter.blur(sigmaX: 80, sigmaY: 80), child: Container(color: Colors.transparent)),
    );
  }

  Widget _legendItem(String label, Color color) {
    return Row(
      children: [
        Container(width: 10, height: 10, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
        const SizedBox(width: 8),
        Text(label, style: const TextStyle(color: Colors.white54, fontSize: 11, fontWeight: FontWeight.bold)),
      ],
    );
  }

  LineChartBarData _lineData(List<FlSpot> spots, Color color) {
    return LineChartBarData(
      spots: spots,
      isCurved: true,
      color: color,
      barWidth: 4,
      isStrokeCapRound: true,
      dotData: FlDotData(show: true, getDotPainter: (s, p, b, i) => FlDotCirclePainter(radius: 4, color: color, strokeWidth: 2, strokeColor: Colors.white)),
      belowBarData: BarAreaData(show: true, gradient: LinearGradient(begin: Alignment.topCenter, end: Alignment.bottomCenter, colors: [color.withOpacity(0.2), color.withOpacity(0.01)])),
    );
  }

  FlTitlesData _barTitles() {
    return FlTitlesData(
      leftTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
      bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, interval: 1, getTitlesWidget: (v, m) => Text("W${v.toInt() + 1}", style: const TextStyle(color: Colors.white24, fontSize: 10)))),
      rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
      topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
    );
  }

  Future<void> _pickDate() async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: selectedDate ?? DateTime.now(),
      firstDate: DateTime(2024),
      lastDate: DateTime.now(),
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: const ColorScheme.dark(
              primary: Colors.blueAccent,
              onPrimary: Colors.white,
              surface: Color(0xFF1E293B),
              onSurface: Colors.white,
            ),
          ),
          child: child!,
        );
      },
    );

    if (picked != null) {
      final auth = Provider.of<AuthProvider>(context, listen: false);
      final dateStr = "${picked.year}-${picked.month.toString().padLeft(2, '0')}-${picked.day.toString().padLeft(2, '0')}";
      
      setState(() => isLoading = true);
      try {
        final result = await ApiConfig.fetchPerformanceByDate(auth.username!, dateStr);
        setState(() {
          selectedDate = picked;
          customWeekData = result['performance'];
          isLoading = false;
        });
      } catch (e) {
        setState(() => isLoading = false);
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error fetching data: $e")));
      }
    }
  }

  Widget _buildRecentSessions() {
    if (history.isEmpty) {
      return _glassCard(
        padding: const EdgeInsets.all(24),
        child: const Center(child: Text("No sessions recorded yet.", style: TextStyle(color: Colors.white38))),
      );
    }

    return Column(
      children: history.take(10).map((session) {
        String cat = session['category'] ?? "N/A";
        String dateStr = session['date'] ?? "N/A";
        if (dateStr.length > 10) dateStr = dateStr.substring(0, 10); // Simple YYYY-MM-DD

        Color color = Colors.blueAccent;
        if (cat == "GD") color = Colors.orangeAccent;
        if (cat == "INTERVIEW") color = Colors.pinkAccent;
        if (cat == "APTITUDE") color = Colors.cyanAccent;

        return Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: InkWell(
            onTap: () => _showReportDetail(session),
            borderRadius: BorderRadius.circular(16),
            child: _glassCard(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(color: color.withOpacity(0.1), shape: BoxShape.circle),
                    child: Icon(_getIconForCategory(cat), color: color, size: 20),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(cat, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)),
                        Text(dateStr, style: const TextStyle(color: Colors.white38, fontSize: 12)),
                      ],
                    ),
                  ),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Text("${session['score']}/${session['total']}", style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w900, fontSize: 18)),
                      Text(session['area'] ?? "", style: const TextStyle(color: Colors.white38, fontSize: 10)),
                    ],
                  ),
                  const SizedBox(width: 12),
                  const Icon(Icons.chevron_right, color: Colors.white24),
                ],
              ),
            ),
          ),
        );
      }).toList(),
    );
  }

  IconData _getIconForCategory(String cat) {
    switch (cat.toUpperCase()) {
      case "GD": return Icons.groups_outlined;
      case "INTERVIEW": return Icons.mic_none;
      case "APTITUDE": return Icons.psychology_outlined;
      case "TECHNICAL": return Icons.code;
      default: return Icons.assignment_outlined;
    }
  }

  void _showReportDetail(Map<String, dynamic> session) {
    showDialog(
      context: context,
      builder: (context) => ReportDetailDialog(
        category: session['category'],
        sessionId: session['id'],
        date: session['date'],
        score: "${session['score']}/${session['total']}",
      ),
    );
  }
}

