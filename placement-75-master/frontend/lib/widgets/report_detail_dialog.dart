import 'dart:convert';
import 'package:flutter/material.dart';
import '../api_config.dart';

class ReportDetailDialog extends StatefulWidget {
  final String category;
  final int sessionId;
  final String date;
  final String score;

  const ReportDetailDialog({
    Key? key,
    required this.category,
    required this.sessionId,
    required this.date,
    required this.score,
  }) : super(key: key);

  @override
  State<ReportDetailDialog> createState() => _ReportDetailDialogState();
}

class _ReportDetailDialogState extends State<ReportDetailDialog> {
  bool loading = true;
  Map<String, dynamic>? details;
  final ScrollController _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _fetchDetails();
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _fetchDetails() async {
    try {
      final data = await ApiConfig.fetchSessionDetail(widget.category, widget.sessionId);
      if (mounted) {
        setState(() {
          details = data;
          loading = false;
        });
      }
    } catch (e) {
      if (mounted) setState(() => loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    String datePart = widget.date;
    if (datePart.length > 16) datePart = datePart.substring(0, 16);

    return Dialog(
      backgroundColor: const Color(0xFF0F172A),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      child: Container(
        width: 600,
        height: 700,
        padding: const EdgeInsets.all(32),
        child: loading
            ? const Center(child: CircularProgressIndicator(color: Colors.indigoAccent))
            : details == null
                ? const Center(child: Text("Failed to load report details.", style: TextStyle(color: Colors.white54)))
                : Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(widget.category, style: const TextStyle(color: Colors.indigoAccent, fontWeight: FontWeight.bold, fontSize: 14)),
                              const SizedBox(height: 4),
                              Text(datePart, style: const TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.w900)),
                            ],
                          ),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                            decoration: BoxDecoration(color: Colors.white.withOpacity(0.05), borderRadius: BorderRadius.circular(12)),
                            child: Text("Score: ${widget.score}", style: const TextStyle(color: Colors.greenAccent, fontWeight: FontWeight.bold, fontSize: 18)),
                          ),
                        ],
                      ),
                      const Divider(height: 40, color: Colors.white12),
                      Expanded(
                        child: Scrollbar(
                          controller: _scrollController,
                          thumbVisibility: true,
                          trackVisibility: true,
                          child: SingleChildScrollView(
                            controller: _scrollController,
                            padding: const EdgeInsets.only(right: 16),
                            child: _buildDetailsContent(),
                          ),
                        ),
                      ),
                      const SizedBox(height: 20),
                      SizedBox(
                        width: double.infinity,
                        child: TextButton(
                          onPressed: () => Navigator.pop(context),
                          child: const Text("Close Report", style: TextStyle(color: Colors.white38)),
                        ),
                      ),
                    ],
                  ),
      ),
    );
  }

  Widget _buildDetailsContent() {
    if (widget.category.toUpperCase() == "GD") {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _styledSection("Topic", details!['topic'] ?? "N/A", color: Colors.indigoAccent),
          const SizedBox(height: 24),
          _styledSection("Your Contribution", details!['transcript'] ?? "N/A"),
          const SizedBox(height: 24),
          _styledSection("AI Feedback", details!['feedback'] ?? "N/A"),
          const SizedBox(height: 24),
          _styledSection("Model Ideal Answer", details!['ideal_answer'] ?? "N/A", color: Colors.greenAccent),
          const SizedBox(height: 32),
          const Text("Score Breakdown", style: TextStyle(color: Colors.white70, fontSize: 13, fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _miniScore("Content", details!['scores']['content']),
              _miniScore("Communication", details!['scores']['communication']),
              _miniScore("Camera/Body", details!['scores']['camera']),
            ],
          ),
        ],
      );
    } else if (widget.category.toUpperCase() == "INTERVIEW") {
      Map<String, dynamic> report = {};
      try {
        if (details!['behavioral_report'] is String) {
          report = json.decode(details!['behavioral_report']);
        } else {
          report = details!['behavioral_report'] ?? {};
        }
      } catch (e) {
        report = {"error": "Could not parse analytics"};
      }

      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _styledSection("Focus Area", details!['area'] ?? 'General Interview', color: Colors.indigoAccent),
          const SizedBox(height: 24),
          const Text("Behavioral Confidence", style: TextStyle(color: Colors.indigoAccent, fontSize: 13, fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          ...report.entries.map((e) => Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Row(
                  children: [
                    Text(e.key.replaceAll('_', ' ').toUpperCase(), style: const TextStyle(color: Colors.white54, fontSize: 10)),
                    const SizedBox(width: 12),
                    Expanded(child: Text(e.value.toString(), style: const TextStyle(color: Colors.white, fontSize: 13))),
                  ],
                ),
              )),
          const Divider(height: 48, color: Colors.white12),
          const Text("Technical Breakdown", style: TextStyle(color: Colors.indigoAccent, fontSize: 14, fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          if (details!['technical_report'] != null && (details!['technical_report'] as List).isNotEmpty)
            ...(details!['technical_report'] as List).map((item) => Padding(
                  padding: const EdgeInsets.only(bottom: 24),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.05),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: Colors.white.withOpacity(0.1))
                        ),
                        child: Text("Q: ${item['question']}", style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 14)),
                      ),
                      const SizedBox(height: 12),
                      Padding(
                        padding: const EdgeInsets.only(left: 8),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text("YOUR ANSWER: ${item['your_answer']}", style: TextStyle(color: Colors.white.withOpacity(0.7), fontSize: 13)),
                            const SizedBox(height: 12),
                            Text("IDEAL ANSWER: ${item['ideal_answer']}", style: const TextStyle(color: Colors.greenAccent, fontSize: 13, fontWeight: FontWeight.bold)),
                            if (item['improvement'] != null) ...[
                              const SizedBox(height: 12),
                              const Text("IMPROVEMENT:", style: TextStyle(color: Colors.orangeAccent, fontSize: 10, fontWeight: FontWeight.bold)),
                              const SizedBox(height: 4),
                              Text(item['improvement'], style: const TextStyle(color: Colors.white70, fontSize: 12, height: 1.4)),
                            ],
                          ],
                        ),
                      ),
                    ],
                  ),
                ))
          else
            const Center(
              child: Padding(
                padding: EdgeInsets.symmetric(vertical: 40),
                child: Text("Detailed technical breakdown only available for sessions after today's update.", 
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.white24, fontSize: 12)),
              ),
            ),
        ],
      );
    } else {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _infoSection("Topic/Area", details!['area'] ?? "General"),
          const SizedBox(height: 24),
          const Text("Quiz Breakdown", style: TextStyle(color: Colors.indigoAccent, fontSize: 14, fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          if (details!['questions'] != null && (details!['questions'] as List).isNotEmpty)
            ...(details!['questions'] as List).map((item) {
              bool isCorrect = item['is_correct'] ?? false;
              bool isSkipped = item['is_skipped'] ?? (item['user_selected'] == null || (item['user_selected'] as String?)?.isEmpty == true);
              String? userSelected = item['user_selected'] as String?;
              List<dynamic> options = item['options'] ?? [];

              // Card color: green = correct, amber = skipped, red = wrong
              Color cardColor = isCorrect
                  ? Colors.green.withOpacity(0.05)
                  : isSkipped
                      ? Colors.amber.withOpacity(0.05)
                      : Colors.red.withOpacity(0.05);
              Color borderColor = isCorrect
                  ? Colors.green.withOpacity(0.25)
                  : isSkipped
                      ? Colors.amber.withOpacity(0.25)
                      : Colors.red.withOpacity(0.25);

              return Padding(
                padding: const EdgeInsets.only(bottom: 24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: cardColor,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: borderColor),
                      ),
                      child: Row(
                        children: [
                          Icon(
                            isCorrect ? Icons.check_circle_outline : isSkipped ? Icons.remove_circle_outline : Icons.cancel_outlined,
                            color: isCorrect ? Colors.greenAccent : isSkipped ? Colors.amberAccent : Colors.redAccent,
                            size: 18,
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text("Q: ${item['question']}", style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 14)),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 12),
                    Padding(
                      padding: const EdgeInsets.only(left: 8),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              const Text("YOU PICKED: ", style: TextStyle(color: Colors.white54, fontSize: 11)),
                              Text(
                                isSkipped ? "Not Selected" : (userSelected ?? "N/A"),
                                style: TextStyle(
                                  color: isSkipped
                                      ? Colors.amberAccent
                                      : isCorrect
                                          ? Colors.greenAccent
                                          : Colors.redAccent,
                                  fontSize: 13,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const SizedBox(width: 20),
                              const Text("CORRECT: ", style: TextStyle(color: Colors.white54, fontSize: 11)),
                              Text(item['correct_answer'] ?? "N/A", style: const TextStyle(color: Colors.greenAccent, fontSize: 13, fontWeight: FontWeight.bold)),
                            ],
                          ),
                          const SizedBox(height: 12),
                          const Text("OPTIONS:", style: TextStyle(color: Colors.white60, fontSize: 10, fontWeight: FontWeight.bold)),
                          const SizedBox(height: 8),
                          ...options.asMap().entries.map((optEntry) {
                            String letter = ["A", "B", "C", "D"][optEntry.key];
                            bool isThisCorrect = letter == item['correct_answer'];
                            // Only highlight as picked (red) if the user actually selected it AND it's wrong
                            bool isThisPicked = !isSkipped && letter == userSelected;

                            return Padding(
                              padding: const EdgeInsets.only(bottom: 4),
                              child: Text(
                                "$letter) ${optEntry.value}",
                                style: TextStyle(
                                  color: isThisCorrect
                                      ? Colors.greenAccent
                                      : isThisPicked
                                          ? Colors.redAccent
                                          : Colors.white38,
                                  fontSize: 12,
                                  fontWeight: (isThisCorrect || isThisPicked) ? FontWeight.bold : FontWeight.normal,
                                ),
                              ),
                            );
                          }),
                          const SizedBox(height: 16),
                          const Text("EXPLANATION:", style: TextStyle(color: Colors.white60, fontSize: 10, fontWeight: FontWeight.bold)),
                          const SizedBox(height: 4),
                          Text(item['explanation'] ?? "No explanation available.", style: const TextStyle(color: Colors.white70, fontSize: 13, height: 1.5)),
                        ],
                      ),
                    ),
                  ],
                ),
              );
            })
          else
            const Center(
              child: Padding(
                padding: EdgeInsets.symmetric(vertical: 40),
                child: Column(
                  children: [
                    Icon(Icons.history_toggle_off, color: Colors.white24, size: 48),
                    SizedBox(height: 16),
                    Text(
                      "No detailed breakdown available.\nOnly sessions completed after today's update show full breakdowns.",
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.white38, fontSize: 13),
                    ),
                  ],
                ),
              ),
            ),
        ],
      );
    }
  }

  Widget _styledSection(String title, String content, {Color color = Colors.white70}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: TextStyle(color: color == Colors.white70 ? Colors.indigoAccent : color, fontSize: 13, fontWeight: FontWeight.bold)),
        const SizedBox(height: 12),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.04),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: Colors.white.withOpacity(0.05)),
          ),
          child: Text(content, style: const TextStyle(color: Colors.white70, height: 1.5, fontSize: 14)),
        ),
      ],
    );
  }

  Widget _infoSection(String title, String content) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: const TextStyle(color: Colors.indigoAccent, fontSize: 14, fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(color: Colors.white.withOpacity(0.03), borderRadius: BorderRadius.circular(16)),
          child: Text(content, style: const TextStyle(color: Colors.white70, height: 1.5, fontSize: 14)),
        ),
      ],
    );
  }

  Widget _miniScore(String label, dynamic score) {
    return Column(
      children: [
        Text(score.toString(), style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold)),
        Text(label, style: const TextStyle(color: Colors.white38, fontSize: 10)),
      ],
    );
  }
}
