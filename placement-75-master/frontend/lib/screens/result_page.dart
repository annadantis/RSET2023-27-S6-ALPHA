import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';
import 'gd_screen.dart';
import 'instructions_screen.dart';

class ResultPage extends StatelessWidget {
  final Map<String, dynamic> result;

  const ResultPage({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    // Extracting data with null safety and type casting
    final double overallScore = (result['overall_score'] ?? 0.0).toDouble();
    final double contentScore = (result['content_score'] ?? 0.0).toDouble();
    final double communicationScore = (result['communication_score'] ?? 0.0).toDouble();
    final double cameraScore = (result['camera_score'] ?? 0.0).toDouble();
    final double voiceScore = (result['voice_score'] ?? 0.0).toDouble();

    final String transcript = result['transcript'] ?? "";
    final String improvedAnswer = result['improved_answer'] ?? "";
    final String idealAnswer = result['ideal_answer'] ?? "";
    final String strategyNote = result['strategy_note'] ?? "";
    final String feedback = result['feedback'] ?? "";
    final String minutesOfMeeting = result['minutes_of_meeting'] ?? "";
    final List<dynamic> responseBreakdown = result['response_breakdown'] ?? [];
    
    // Parsing the content audit for specific metrics
    final Map<String, dynamic> audit = result['content_audit'] is String 
        ? jsonDecode(result['content_audit']) 
        : (result['content_audit'] ?? {});
    
    final int wpm = audit['wpm'] ?? 0;
    final int fillersCount = audit['filler_words_count'] ?? 0;

    Color getScoreColor(double score) {
      if (score >= 7.5) return Colors.greenAccent;
      if (score >= 5.0) return Colors.yellowAccent;
      return Colors.redAccent;
    }

    return Scaffold(
      backgroundColor: const Color(0xFF0F0C29),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text(
          "Performance Analysis",
          style: GoogleFonts.inter(fontWeight: FontWeight.bold, color: Colors.white),
        ),
        centerTitle: true,
        leading: IconButton(
          icon: const Icon(Icons.close, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Overall Readiness Gauge
            Center(
              child: CircularPercentIndicator(
                radius: 85.0,
                lineWidth: 14.0,
                percent: overallScore / 10.0,
                center: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      overallScore.toStringAsFixed(1),
                      style: GoogleFonts.poppins(
                        fontSize: 36,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    Text(
                      "OVERALL",
                      style: GoogleFonts.inter(
                        fontSize: 10,
                        color: Colors.white60,
                        letterSpacing: 2,
                      ),
                    ),
                  ],
                ),
                circularStrokeCap: CircularStrokeCap.round,
                backgroundColor: Colors.white10,
                progressColor: getScoreColor(overallScore),
                animation: true,
                animationDuration: 1200,
              ),
            ),

            const SizedBox(height: 32),

            // Performance Pillars - Single Row
            Row(
              children: [
                Expanded(child: _buildPillarCard("Content", contentScore.toStringAsFixed(1), Icons.psychology, Colors.blueAccent)),
                const SizedBox(width: 8),
                Expanded(child: _buildPillarCard("Comm.", communicationScore.toStringAsFixed(1), Icons.forum, Colors.purpleAccent)),
                const SizedBox(width: 8),
                Expanded(child: _buildPillarCard("Visual", cameraScore.toStringAsFixed(1), Icons.visibility, Colors.orangeAccent)),
              ],
            ),

            const SizedBox(height: 32),
            // Metrics bar removed per request

            const SizedBox(height: 32),

            const SizedBox(height: 32),

            // Transcript Section
            Text(
              "Your Transcript",
              style: GoogleFonts.inter(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white),
            ),
            const SizedBox(height: 16),
            _buildComparisonBox("WHAT YOU SAID", transcript, Colors.white70, isItalic: true),

            if (strategyNote.isNotEmpty) ...[
              const SizedBox(height: 16),
            ],

            const SizedBox(height: 32),

            // ── MINUTES OF MEETING ──────────────────────────────────────
            if (minutesOfMeeting.isNotEmpty) ...[
              Text(
                "Minutes of Meeting",
                style: GoogleFonts.inter(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white),
              ),
              const SizedBox(height: 4),
              Text(
                "Full conversation during the GD session",
                style: GoogleFonts.inter(fontSize: 12, color: Colors.white38),
              ),
              const SizedBox(height: 12),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: const Color(0xFF161625),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: Colors.white10),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: minutesOfMeeting.split('\n').where((l) => l.trim().isNotEmpty).map((line) {
                    final isUser = line.toLowerCase().startsWith('you:') || line.toLowerCase().startsWith('user:');
                    final isThomas = line.toLowerCase().startsWith('thomas:');
                    final isAravind = line.toLowerCase().startsWith('aravind:');
                    Color lineColor = Colors.white60;
                    if (isUser) lineColor = Colors.cyanAccent;
                    else if (isThomas) lineColor = Colors.amberAccent;
                    else if (isAravind) lineColor = Colors.greenAccent;
                    else lineColor = Colors.pinkAccent; // George
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Text(
                        line,
                        style: GoogleFonts.inter(fontSize: 13, color: lineColor, height: 1.4),
                      ),
                    );
                  }).toList(),
                ),
              ),
              const SizedBox(height: 32),
            ],

            // ── PER-RESPONSE BREAKDOWN ──────────────────────────────────
            if (responseBreakdown.isNotEmpty) ...[
              Text(
                "Your Response Analysis",
                style: GoogleFonts.inter(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white),
              ),
              const SizedBox(height: 4),
              Text(
                "In-depth analysis and ideal benchmarks",
                style: GoogleFonts.inter(fontSize: 12, color: Colors.white38),
              ),
              const SizedBox(height: 12),
              
              const SizedBox(height: 12),

              ...responseBreakdown.asMap().entries.map((entry) {
                final item = entry.value as Map<String, dynamic>;
                final turnNum = item['turn'] ?? (entry.key + 1);
                return Container(
                  margin: const EdgeInsets.only(bottom: 16),
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.04),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: Colors.white10),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        "Turn $turnNum",
                        style: GoogleFonts.inter(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.cyanAccent, letterSpacing: 1),
                      ),
                      const SizedBox(height: 8),
                      _buildComparisonBox("YOU SAID", (item['response'] ?? "").toString(), Colors.white60, isItalic: true),
                      if ((item['ideal'] ?? "").toString().isNotEmpty) ...[
                        const SizedBox(height: 12),
                        _buildIdealAnswerWidget((item['ideal'] ?? "").toString(), title: "MASTER IDEAL ANSWER"),
                      ],
                      if ((item['feedback'] ?? "").toString().isNotEmpty) ...[
                        const SizedBox(height: 8),
                        _buildComparisonBox("FEEDBACK EVALUATION", (item['feedback'] ?? "").toString(), Colors.amberAccent),
                      ],
                    ],
                  ),
                );
              }),
              const SizedBox(height: 16),
            ],

            // ── OVERALL AI COACHING FEEDBACK ───────────────────────────
            _buildFeedbackCard(feedback),

            const SizedBox(height: 40),
            
            SizedBox(
              width: double.infinity,
              height: 55,
              child: ElevatedButton(
                onPressed: () => Navigator.pushAndRemoveUntil(
                  context,
                  MaterialPageRoute(
                    builder: (_) => InstructionsScreen(
                      category: "GD",
                      onStart: () {
                        Navigator.pushReplacement(context, MaterialPageRoute(builder: (context) => const GdScreen()));
                      },
                    ),
                  ),
                  (route) => route.isFirst,
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.cyanAccent,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                ),
                child: const Text("PRACTICE AGAIN", style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),
              ),
            ),
            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }

  Widget _buildPillarCard(String title, String score, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFF161625),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.1)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: color, size: 16),
          const SizedBox(width: 6),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(score, style: GoogleFonts.poppins(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.white)),
                Text(title, style: GoogleFonts.inter(fontSize: 8, color: Colors.white54, letterSpacing: 0.5)),
              ],
            ),
          )
        ],
      ),
    );
  }

  // _buildMetricsBar removed per request

  Widget _metricItem(String label, String value, Color color) {
    return Column(
      children: [
        Text(value, style: GoogleFonts.poppins(fontSize: 20, fontWeight: FontWeight.bold, color: color)),
        Text(label, style: GoogleFonts.inter(fontSize: 10, color: Colors.white38, letterSpacing: 1)),
      ],
    );
  }

  Widget _buildComparisonBox(String label, String content, Color textColor, {bool isItalic = false}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: GoogleFonts.inter(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.white38, letterSpacing: 1)),
        const SizedBox(height: 8),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: const Color(0xFF161625),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Text(
            content.isEmpty ? "No data available." : content,
            style: GoogleFonts.inter(
              fontSize: 14, 
              color: textColor, 
              height: 1.5,
              fontStyle: isItalic ? FontStyle.italic : FontStyle.normal,
            ),
          ),
        ),
      ],
    );
  }

 

  Widget _buildIdealAnswerWidget(String content, {required String title}) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            const Color(0xFF1D976C).withOpacity(0.15),
            const Color(0xFF1D976C).withOpacity(0.05),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFF1D976C).withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.auto_awesome, color: Color(0xFF1D976C), size: 18),
              const SizedBox(width: 8),
              Text(
                title,
                style: GoogleFonts.inter(
                  fontSize: 11, 
                  fontWeight: FontWeight.bold, 
                  color: const Color(0xFF1D976C),
                  letterSpacing: 0.5,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            content,
            style: GoogleFonts.inter(
              fontSize: 14,
              color: Colors.white,
              height: 1.6,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFeedbackCard(String feedback) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1A2980).withOpacity(0.2),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.blueAccent.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text("DETAILED FEEDBACK", style: GoogleFonts.inter(fontWeight: FontWeight.bold, color: Colors.blueAccent, fontSize: 12)),
          const SizedBox(height: 12),
          Text(feedback, style: GoogleFonts.inter(fontSize: 14, color: Colors.white, height: 1.6)),
        ],
      ),
    );
  }
}





