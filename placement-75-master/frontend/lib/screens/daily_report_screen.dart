import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:intl/intl.dart';
import '../providers/auth_provider.dart';
import '../api_config.dart';
import '../widgets/report_detail_dialog.dart';

class DailyReportScreen extends StatefulWidget {
  const DailyReportScreen({Key? key}) : super(key: key);

  @override
  _DailyReportScreenState createState() => _DailyReportScreenState();
}

class _DailyReportScreenState extends State<DailyReportScreen> {
  DateTime selectedDate = DateTime.now();
  Map<String, dynamic>? reportData;
  bool isLoading = false;
  String errorMessage = '';

  @override
  void initState() {
    super.initState();
    _fetchDailyReport();
  }

  Future<void> _fetchDailyReport() async {
    final auth = Provider.of<AuthProvider>(context, listen: false);
    if (auth.username == null) return;

    setState(() {
      isLoading = true;
      errorMessage = '';
      reportData = null;
    });

    final dateStr = DateFormat('yyyy-MM-dd').format(selectedDate);
    final url = Uri.parse('${ApiConfig.baseUrl}/daily_report/${auth.username}?date_str=$dateStr');

    try {
      final response = await http.get(url);
      if (response.statusCode == 200) {
        setState(() {
          reportData = json.decode(response.body);
          isLoading = false;
        });
      } else {
        setState(() {
          errorMessage = 'Failed to load report. Status code: ${response.statusCode}';
          isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        errorMessage = 'Network error: $e';
        isLoading = false;
      });
    }
  }

  Future<void> _selectDate(BuildContext context) async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: selectedDate,
      firstDate: DateTime(2020),
      lastDate: DateTime.now(),
      builder: (context, child) {
        return Theme(
          data: ThemeData.dark().copyWith(
            colorScheme: const ColorScheme.dark(
              primary: Colors.blueAccent,
              onPrimary: Colors.white,
              surface: Color(0xFF161625),
              onSurface: Colors.white,
            ),
          ),
          child: child!,
        );
      },
    );
    if (picked != null && picked != selectedDate) {
      setState(() {
        selectedDate = picked;
      });
      _fetchDailyReport();
    }
  }

  Widget _buildEmptyState(String message) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.inbox_outlined, size: 60, color: Colors.white24),
          const SizedBox(height: 16),
          Text(message, style: TextStyle(color: Colors.white54, fontSize: 16)),
        ],
      ),
    );
  }

  Widget _buildResultCard({required String title, required List<Widget> children}) {
    return Container(
      margin: const EdgeInsets.only(bottom: 20),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF161625),
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(color: Colors.blueAccent, fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const Divider(color: Colors.white10, height: 30),
          ...children,
        ],
      ),
    );
  }

  Widget _buildQuizReport(String category, List<dynamic> results) {
    if (results.isEmpty) {
      return _buildResultCard(
        title: category,
        children: [const Text("No sessions recorded for this day.", style: TextStyle(color: Colors.white54))],
      );
    }

    return _buildResultCard(
      title: category,
      children: results.map((res) {
        return InkWell(
          onTap: () {
            showDialog(
              context: context,
              builder: (context) => ReportDetailDialog(
                category: category.contains("Technical") ? "TECHNICAL" : "APTITUDE",
                sessionId: res['id'],
                date: res['timestamp'] ?? DateFormat('yyyy-MM-dd').format(selectedDate),
                score: "${res['score']}/${res['total_questions']}",
              ),
            );
          },
          child: Padding(
            padding: const EdgeInsets.only(bottom: 15),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("Area: ${res['area'] ?? 'General'}", style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
                      Text("Time: ${res['timestamp']?.toString().split(' ').last.split('.').first ?? ''}", style: const TextStyle(color: Colors.white38, fontSize: 12)),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.blueAccent.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: Colors.blueAccent.withOpacity(0.5)),
                  ),
                  child: Text(
                    "${res['score']}/${res['total_questions']}",
                    style: const TextStyle(color: Colors.blueAccent, fontWeight: FontWeight.bold),
                  ),
                ),
                const SizedBox(width: 8),
                const Icon(Icons.chevron_right, color: Colors.white24, size: 16),
              ],
            ),
          ),
        );
      }).toList(),
    );
  }

  Widget _buildInterviewReport(List<dynamic> results) {
    if (results.isEmpty) {
      return _buildResultCard(
        title: "Interview Practice",
        children: [const Text("No interview practice recorded for this day.", style: TextStyle(color: Colors.white54))],
      );
    }
    
    return _buildResultCard(
      title: "Interview Practice",
      children: results.map((res) {
        return InkWell(
          onTap: () {
            showDialog(
              context: context,
              builder: (context) => ReportDetailDialog(
                category: "INTERVIEW",
                sessionId: res['id'],
                date: res['timestamp'] ?? DateFormat('yyyy-MM-dd').format(selectedDate),
                score: "${res['score'] ?? 0}/10",
              ),
            );
          },
          child: Padding(
            padding: const EdgeInsets.only(bottom: 15),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text("Interview Session", style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
                      if (res['confidence'] != null && res['confidence'].toString().isNotEmpty)
                        Text(
                          "Analysis: ${res['confidence']}",
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(color: Colors.white54, fontSize: 12),
                        ),
                    ],
                  ),
                ),
                Row(
                  children: [
                    Text("${res['score'] ?? 0}/10", style: const TextStyle(color: Colors.greenAccent, fontWeight: FontWeight.bold, fontSize: 16)),
                    const SizedBox(width: 8),
                    const Icon(Icons.chevron_right, color: Colors.white24, size: 16),
                  ],
                ),
              ],
            ),
          ),
        );
      }).toList(),
    );
  }

  Widget _buildGDReport(List<dynamic> results) {
    if (results.isEmpty) {
      return _buildResultCard(
        title: "Group Discussion",
        children: [const Text("No GD sessions recorded for this day.", style: TextStyle(color: Colors.white54))],
      );
    }

    return _buildResultCard(
      title: "Group Discussion",
      children: results.map((res) {
        return InkWell(
          onTap: () {
            showDialog(
              context: context,
              builder: (context) => ReportDetailDialog(
                category: "GD",
                sessionId: res['id'],
                date: res['timestamp'] ?? DateFormat('yyyy-MM-dd').format(selectedDate),
                score: "${res['final_score'] ?? 0}/100",
              ),
            );
          },
          child: Padding(
            padding: const EdgeInsets.only(bottom: 15),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text("GD Session", style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
                          Text("Time: ${res['timestamp']?.toString().split(' ').last.split('.').first ?? ''}", style: const TextStyle(color: Colors.white38, fontSize: 12)),
                        ],
                      ),
                    ),
                    Row(
                      children: [
                        Text("${res['final_score'] ?? 0}/100", style: const TextStyle(color: Colors.orangeAccent, fontWeight: FontWeight.bold)),
                        const SizedBox(width: 8),
                        const Icon(Icons.chevron_right, color: Colors.white24, size: 16),
                      ],
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Row(
                  children: [
                    _miniStat("Content", "${res['content_score'] ?? 0}/10"),
                    const SizedBox(width: 15),
                    _miniStat("Communication", "${res['communication_score'] ?? 0}/10"),
                  ],
                )
              ],
            ),
          ),
        );
      }).toList(),
    );
  }

  Widget _miniStat(String label, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(color: Colors.white38, fontSize: 12)),
        const SizedBox(height: 4),
        Text(value, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent, // Let dashboard color show through
      body: Padding(
        padding: const EdgeInsets.all(30.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header & Date Picker
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      "Report",
                      style: TextStyle(color: Colors.white, fontSize: 28, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 5),
                    Text(
                      "Review your performance and progress",
                      style: TextStyle(color: Colors.white54, fontSize: 14),
                    ),
                  ],
                ),
                InkWell(
                  onTap: () => _selectDate(context),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                    decoration: BoxDecoration(
                      color: const Color(0xFF161625),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.white10),
                    ),
                    child: Row(
                      children: [
                        const Icon(Icons.calendar_today, color: Colors.blueAccent, size: 18),
                        const SizedBox(width: 10),
                        Text(
                          DateFormat('MMM dd, yyyy').format(selectedDate),
                          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
                        ),
                        const SizedBox(width: 5),
                        const Icon(Icons.arrow_drop_down, color: Colors.white54),
                      ],
                    ),
                  ),
                )
              ],
            ),
            const SizedBox(height: 40),
            
            // Content
            Expanded(
              child: isLoading
                  ? const Center(child: CircularProgressIndicator(color: Colors.blueAccent))
                  : errorMessage.isNotEmpty
                      ? _buildEmptyState(errorMessage)
                      : reportData == null
                          ? _buildEmptyState("Please select a date to view report")
                          : ListView(
                              children: [
                                _buildQuizReport("Technical Practice", reportData!['TECHNICAL'] ?? []),
                                _buildQuizReport("Aptitude Practice", reportData!['APTITUDE'] ?? []),
                                _buildInterviewReport(reportData!['INTERVIEW'] ?? []),
                                _buildGDReport(reportData!['GD'] ?? []),
                              ],
                            ),
            )
          ],
        ),
      ),
    );
  }
}
