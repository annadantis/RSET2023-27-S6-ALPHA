import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:provider/provider.dart';
import 'dart:convert';
import 'dart:async';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:placement_assistant/widgets/loading_overlay.dart';
import '../providers/auth_provider.dart';
import '../api_config.dart';

// Question status enum
enum QuestionStatus { unattempted, attempted, markedForReview, attemptedAndMarked }

class QuizScreen extends StatefulWidget {
  final String category;
  final String? targetBranch; // Added for practice mode
  const QuizScreen({super.key, required this.category, this.targetBranch});

  @override
  State<QuizScreen> createState() => _QuizScreenState();
}

class _QuizScreenState extends State<QuizScreen> {
  List<dynamic> questions = [];
  int currentIndex = 0;
  int score = 0;
  bool loading = true;
  String selectedOption = "";
  final FlutterTts flutterTts = FlutterTts();

  // Timer variables
  Timer? _timer;
  int _timeLeft = 300; // 5 minutes for 10 questions

  // Per-question state
  List<String> _selectedAnswers = []; // user's selected option per question
  List<bool> _showExplanation = [];   // whether to show explanation per question
  List<QuestionStatus> _status = [];  // status for each question

  // Theme Colors
  final Color scaffoldBg = const Color(0xFF0F0C29);
  final Color cardBg = const Color(0xFF161625);
  final Color accentColor = const Color(0xFF2196F3);

  @override
  void initState() {
    super.initState();
    _initTts();
    fetchQuestions();
  }

  void _initTts() async {
    await flutterTts.setLanguage("en-US");
    await flutterTts.setSpeechRate(0.5);
    await flutterTts.setVolume(1.0);
  }

  void _startTimer() {
    if (_timer != null) return; // Only start once
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (_timeLeft > 0) {
        if (mounted) {
          setState(() {
            _timeLeft--;
          });
        }
      } else {
        _timer?.cancel();
        _completeQuiz(); // Global time up, finish quiz
      }
    });
  }

  Future<void> _readAloud() async {
    if (questions.isNotEmpty && currentIndex < questions.length) {
      var q = questions[currentIndex];
      final opts = q['options'];
      List<dynamic> optionsList = opts is List ? opts : (opts is Map ? opts.values.toList() : []);
      String speech = "Question ${currentIndex + 1}: ${q['question']}.";
      for (int i = 0; i < optionsList.length && i < 4; i++) {
        speech += " Option ${['A', 'B', 'C', 'D'][i]}: ${optionsList[i]}.";
      }
      await flutterTts.speak(speech);
    }
  }

  @override
  void dispose() {
    flutterTts.stop();
    _timer?.cancel();
    super.dispose();
  }

  Future<void> fetchQuestions() async {
    final auth = Provider.of<AuthProvider>(context, listen: false);
    final url = Uri.parse('${auth.baseUrl}/get_daily_quiz');
    try {
      final res = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          "username": auth.username,
          "category": widget.category,
          "target_branch": widget.targetBranch
        }),
      );
      debugPrint("API Response: ${res.body}");
      if (res.statusCode == 200) {
        if (mounted) {
          final qs = jsonDecode(res.body)['questions'] as List;
          setState(() {
            questions = qs;
            _selectedAnswers = List.filled(qs.length, "");
            _showExplanation = List.filled(qs.length, false);
            _status = List.filled(qs.length, QuestionStatus.unattempted);
            loading = false;
          });
          _readAloud();
          _startTimer();
        }
      } else {
        debugPrint("Fetch Error: ${res.statusCode} ${res.body}");
        if (mounted) setState(() => loading = false);
      }
    } catch (e) {
      debugPrint("Fetch Error: $e");
      if (mounted) setState(() => loading = false);
    }
  }

  void handleAnswer(String opt) async {
    if (_showExplanation[currentIndex]) return; // already answered
    flutterTts.stop();

    setState(() {
      _selectedAnswers[currentIndex] = opt;
      _status[currentIndex] = _status[currentIndex] == QuestionStatus.markedForReview
          ? QuestionStatus.attemptedAndMarked
          : QuestionStatus.attempted;
    });

    final auth = Provider.of<AuthProvider>(context, listen: false);
    final url = Uri.parse('${auth.baseUrl}/check_answer');

    try {
      final res = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          "username": auth.username,
          "category": widget.category,
          "question_id": questions[currentIndex]['id'],
          "user_answer": opt
        }),
      );

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        if (mounted) {
          setState(() {
            _showExplanation[currentIndex] = true;
            questions[currentIndex]['explanation'] = data['explanation'];
            questions[currentIndex]['user_selected'] = opt;
            if (data['is_correct']) score++;
          });
        }
      }
    } catch (e) {
      debugPrint("Error checking answer: $e");
      if (mounted) {
        setState(() {
          _showExplanation[currentIndex] = true;
          questions[currentIndex]['user_selected'] = opt;
          if (opt == questions[currentIndex]['answer']) score++;
        });
      }
    }
  }

  void _toggleMarkForReview() {
    setState(() {
      if (_status[currentIndex] == QuestionStatus.unattempted) {
        _status[currentIndex] = QuestionStatus.markedForReview;
      } else if (_status[currentIndex] == QuestionStatus.markedForReview) {
        _status[currentIndex] = QuestionStatus.unattempted;
      } else if (_status[currentIndex] == QuestionStatus.attempted) {
        _status[currentIndex] = QuestionStatus.attemptedAndMarked;
      } else if (_status[currentIndex] == QuestionStatus.attemptedAndMarked) {
        _status[currentIndex] = QuestionStatus.attempted;
      }
    });
  }

  void nextQuestion() async {
    if (currentIndex < questions.length - 1) {
      setState(() {
        currentIndex++;
      });
      _readAloud();
    } else {
      _completeQuiz();
    }
  }

  void _goToQuestion(int index) {
    setState(() => currentIndex = index);
    _readAloud();
  }

  Future<void> _completeQuiz() async {
    _timer?.cancel();
    final auth = Provider.of<AuthProvider>(context, listen: false);

    Map<String, int> areaMistakes = {};
    List<Map<String, dynamic>> finalAnswers = [];

    if (questions.isNotEmpty) {
       for (var q in questions) {
         bool correct = q['user_selected'] == q['answer'];
         if (!correct) {
           String area = q['area'] ?? "General";
           areaMistakes[area] = (areaMistakes[area] ?? 0) + 1;
         }
         finalAnswers.add({
           "question_id": q['id'],
           "user_answer": q['user_selected'] ?? "",
           "is_correct": correct ? 1 : 0
         });
       }
    }

    String weakArea = "General";
    if (areaMistakes.isNotEmpty) {
      var sortedKeys = areaMistakes.keys.toList(growable: false)
        ..sort((k1, k2) => areaMistakes[k2]!.compareTo(areaMistakes[k1]!));
      weakArea = sortedKeys.first;
    }

    final res = await http.post(
      Uri.parse('${auth.baseUrl}/submit_quiz'),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({
        "username": auth.username,
        "category": widget.category,
        "score": score,
        "total_questions": questions.length,
        "target_branch": widget.targetBranch,
        "weak_area": weakArea,
        "answers": finalAnswers
      }),
    );

    bool isLevelUp = false;
    if (res.statusCode == 200) {
      isLevelUp = jsonDecode(res.body)['level_up'] ?? false;
    }

    _showResultDialog(isLevelUp, weakArea);
  }

  void _showResultDialog(bool leveledUp, String weakArea) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => AlertDialog(
        backgroundColor: cardBg,
        title: Text(leveledUp ? "🎉 Level Up!" : "Quiz Over", style: const TextStyle(color: Colors.white)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text("You scored $score/${questions.length}.\n", style: const TextStyle(color: Colors.white70, fontSize: 18)),
            if (score < questions.length * 0.7)
              Text("Weak Area: $weakArea", style: const TextStyle(color: Colors.redAccent, fontWeight: FontWeight.bold, fontSize: 18)),
            const SizedBox(height: 12),
            Text(
                (leveledUp
                  ? 'You have reached a new difficulty level!'
                  : (widget.category == "TECHNICAL" && widget.targetBranch != null)
                    ? "Practice mode complete."
                    : 'Try to score 70%+ to level up.'),
                style: const TextStyle(color: Colors.white70, fontSize: 16)),
          ],
        ),
        actions: [
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: accentColor),
            onPressed: () => Navigator.popUntil(context, (r) => r.isFirst),
            child: const Text("Done", style: TextStyle(color: Colors.white)),
          )
        ],
      ),
    );
  }

  Future<bool> _onWillPop() async {
    final answered = _selectedAnswers.where((s) => s.isNotEmpty).length;
    if (answered == 0 && score == 0) return true; // Nothing done, safe to exit

    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: cardBg,
        title: const Text("Leave Quiz?", style: TextStyle(color: Colors.white)),
        content: const Text(
          "⚠️ Your current quiz progress will NOT be saved. The test result will not be recorded in your analytics.\n\nAre you sure you want to leave?",
          style: TextStyle(color: Colors.white70, height: 1.5),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text("Continue Quiz", style: TextStyle(color: Colors.blueAccent)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text("Leave", style: TextStyle(color: Colors.redAccent)),
          ),
        ],
      ),
    );
    if (confirm == true) _timer?.cancel();
    return confirm ?? false;
  }

  // ========================= BUILD =========================

  @override
  Widget build(BuildContext context) {
    if (loading) return PremiumLoadingOverlay(message: "Generating Adaptive Quiz Questions...");
    if (questions.isEmpty) return Scaffold(backgroundColor: scaffoldBg, body: const Center(child: Text("No questions found.", style: TextStyle(color: Colors.white))));

    return WillPopScope(
      onWillPop: _onWillPop,
      child: Scaffold(
        backgroundColor: scaffoldBg,
        body: Row(
          children: [
            _buildQuestionSidebar(),
            Expanded(child: _buildQuizContent()),
          ],
        ),
      ),
    );
  }

  // =========== SIDEBAR ===========

  Widget _buildQuestionSidebar() {
    final attempted = _status.where((s) => s == QuestionStatus.attempted || s == QuestionStatus.attemptedAndMarked).length;
    final left = _status.where((s) => s == QuestionStatus.unattempted || s == QuestionStatus.markedForReview).length;
    final marked = _status.where((s) => s == QuestionStatus.markedForReview || s == QuestionStatus.attemptedAndMarked).length;

    return Container(
      width: 200,
      color: cardBg,
      child: Column(
        children: [
          const SizedBox(height: 40),
          // Timer in sidebar
          Container(
            margin: const EdgeInsets.symmetric(horizontal: 12),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            decoration: BoxDecoration(
              color: _timeLeft < 60
                  ? Colors.red.withOpacity(0.2)
                  : Colors.blue.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: _timeLeft < 60 ? Colors.redAccent : Colors.blueAccent.withOpacity(0.3),
              ),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.timer_outlined,
                    color: _timeLeft < 60 ? Colors.redAccent : Colors.blueAccent,
                    size: 18),
                const SizedBox(width: 6),
                Text(
                  "${(_timeLeft ~/ 60).toString().padLeft(2, '0')}:${(_timeLeft % 60).toString().padLeft(2, '0')}",
                  style: TextStyle(
                    color: _timeLeft < 60 ? Colors.redAccent : Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 20,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // Summary stats
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Column(
              children: [
                _statRow(Icons.check_circle_outline, "Answered", "$attempted", Colors.greenAccent),
                const SizedBox(height: 6),
                _statRow(Icons.radio_button_unchecked, "Left", "$left", Colors.white54),
                const SizedBox(height: 6),
                _statRow(Icons.bookmark_outline, "Marked", "$marked", Colors.amberAccent),
              ],
            ),
          ),
          const Divider(color: Colors.white10, height: 24),

          // Question grid label
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 12),
            child: Align(
              alignment: Alignment.centerLeft,
              child: Text("Questions", style: TextStyle(color: Colors.white54, fontSize: 11, fontWeight: FontWeight.bold, letterSpacing: 1.2)),
            ),
          ),
          const SizedBox(height: 8),

          // Question number grid
          Expanded(
            child: GridView.builder(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 3,
                crossAxisSpacing: 8,
                mainAxisSpacing: 8,
                childAspectRatio: 1.1,
              ),
              itemCount: questions.length,
              itemBuilder: (ctx, i) => _questionBubble(i),
            ),
          ),

          // Legend
          Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Divider(color: Colors.white10),
                const SizedBox(height: 4),
                _legendRow(Colors.greenAccent, "Answered"),
                _legendRow(Colors.amberAccent, "Marked"),
                _legendRow(Colors.indigoAccent, "Current"),
                _legendRow(Colors.white24, "Not Visited"),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _statRow(IconData icon, String label, String value, Color color) {
    return Row(
      children: [
        Icon(icon, color: color, size: 14),
        const SizedBox(width: 6),
        Text(label, style: const TextStyle(color: Colors.white54, fontSize: 11)),
        const Spacer(),
        Text(value, style: TextStyle(color: color, fontWeight: FontWeight.bold, fontSize: 13)),
      ],
    );
  }

  Widget _questionBubble(int index) {
    final isCurrent = index == currentIndex;
    final status = _status[index];

    Color bgColor;
    Color borderColor;
    Color textColor = Colors.white;

    if (isCurrent) {
      bgColor = Colors.indigoAccent.withOpacity(0.4);
      borderColor = Colors.indigoAccent;
    } else if (status == QuestionStatus.attempted) {
      bgColor = Colors.greenAccent.withOpacity(0.15);
      borderColor = Colors.greenAccent;
    } else if (status == QuestionStatus.markedForReview) {
      bgColor = Colors.amber.withOpacity(0.15);
      borderColor = Colors.amberAccent;
    } else if (status == QuestionStatus.attemptedAndMarked) {
      bgColor = Colors.green.withOpacity(0.1);
      borderColor = Colors.amberAccent;
    } else {
      bgColor = Colors.white.withOpacity(0.04);
      borderColor = Colors.white24;
      textColor = Colors.white54;
    }

    return GestureDetector(
      onTap: () => _goToQuestion(index),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        decoration: BoxDecoration(
          color: bgColor,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: borderColor, width: isCurrent ? 2 : 1),
        ),
        child: Stack(
          alignment: Alignment.center,
          children: [
            Text(
              "${index + 1}",
              style: TextStyle(color: isCurrent ? Colors.white : textColor, fontWeight: FontWeight.bold, fontSize: 14),
            ),
            if (status == QuestionStatus.markedForReview || status == QuestionStatus.attemptedAndMarked)
              Positioned(
                top: 2,
                right: 3,
                child: Icon(Icons.bookmark, color: Colors.amberAccent, size: 10),
              ),
          ],
        ),
      ),
    );
  }

  Widget _legendRow(Color color, String label) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        children: [
          Container(width: 10, height: 10, decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(3))),
          const SizedBox(width: 6),
          Text(label, style: const TextStyle(color: Colors.white38, fontSize: 10)),
        ],
      ),
    );
  }

  // =========== QUIZ CONTENT ===========

  Widget _buildQuizContent() {
    final q = questions[currentIndex];
    final options = q['options'];
    List<dynamic> optionsList = options is List ? options : (options is Map ? options.values.toList() : []);

    final isAnswered = _showExplanation[currentIndex];
    final selectedOption = _selectedAnswers[currentIndex];
    final isMarked = _status[currentIndex] == QuestionStatus.markedForReview ||
        _status[currentIndex] == QuestionStatus.attemptedAndMarked;

    return Column(
      children: [
        // Top bar
        Container(
          padding: const EdgeInsets.fromLTRB(16, 40, 16, 12),
          color: cardBg,
          child: Row(
            children: [
              IconButton(
                icon: const Icon(Icons.close, color: Colors.white70),
                onPressed: () async {
                  if (await _onWillPop()) Navigator.pop(context);
                },
              ),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      "${widget.category} Quiz",
                      style: TextStyle(color: accentColor, fontWeight: FontWeight.bold, fontSize: 12),
                    ),
                    Text(
                      "Question ${currentIndex + 1} of ${questions.length}",
                      style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16),
                    ),
                  ],
                ),
              ),
              // Mark for review button
              IconButton(
                icon: Icon(
                  isMarked ? Icons.bookmark : Icons.bookmark_border,
                  color: isMarked ? Colors.amberAccent : Colors.white38,
                ),
                onPressed: _toggleMarkForReview,
                tooltip: isMarked ? "Unmark Review" : "Mark for Review",
              ),
              IconButton(
                icon: const Icon(Icons.volume_up, color: Colors.white38),
                onPressed: _readAloud,
              ),
            ],
          ),
        ),

        // Progress bar
        LinearProgressIndicator(
          value: (currentIndex + 1) / questions.length,
          minHeight: 4,
          backgroundColor: Colors.white10,
          color: accentColor,
        ),

        // Question body
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Difficulty & area tags
                Wrap(
                  spacing: 8,
                  children: [
                    _tagChip(q['difficulty'] ?? 'Easy', accentColor),
                    _tagChip(q['area'] ?? 'General', Colors.white30),
                  ],
                ),
                const SizedBox(height: 20),

                Text(q['question'], style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w600, color: Colors.white, height: 1.5)),
                const SizedBox(height: 30),

                // Options
                ...optionsList.asMap().entries.map((entry) {
                  int idx = entry.key;
                  String opt = entry.value;
                  String optionLetter = ["A", "B", "C", "D"][idx];
                  String correctAns = (q['answer'] ?? "").toString().trim().toUpperCase();
                  String mySelected = selectedOption.trim().toUpperCase();

                  bool isCorrect = optionLetter == correctAns;
                  bool isSelected = optionLetter == mySelected;

                  Color btnColor = cardBg;
                  Color borderCol = Colors.white12;
                  Color textCol = Colors.white;

                  if (isAnswered) {
                    if (isCorrect) {
                      btnColor = Colors.green.withOpacity(0.2);
                      borderCol = Colors.greenAccent;
                      textCol = Colors.greenAccent;
                    } else if (isSelected) {
                      btnColor = Colors.red.withOpacity(0.2);
                      borderCol = Colors.redAccent;
                      textCol = Colors.redAccent;
                    }
                  } else if (isSelected) {
                    btnColor = accentColor.withOpacity(0.15);
                    borderCol = accentColor;
                    textCol = accentColor;
                  }

                  return Padding(
                    padding: const EdgeInsets.only(bottom: 14),
                    child: InkWell(
                      onTap: isAnswered ? null : () => handleAnswer(optionLetter),
                      borderRadius: BorderRadius.circular(14),
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 250),
                        width: double.infinity,
                        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
                        decoration: BoxDecoration(
                          color: btnColor,
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(color: borderCol, width: isAnswered && isCorrect ? 2 : 1),
                        ),
                        child: Row(
                          children: [
                            Container(
                              width: 28,
                              height: 28,
                              decoration: BoxDecoration(
                                color: borderCol.withOpacity(0.15),
                                shape: BoxShape.circle,
                                border: Border.all(color: borderCol),
                              ),
                              child: Center(child: Text(optionLetter, style: TextStyle(color: borderCol, fontWeight: FontWeight.bold, fontSize: 12))),
                            ),
                            const SizedBox(width: 14),
                            Expanded(child: Text(opt, style: TextStyle(color: textCol, fontSize: 15))),
                            if (isAnswered && isCorrect) const Icon(Icons.check_circle, color: Colors.greenAccent, size: 20),
                            if (isAnswered && isSelected && !isCorrect) const Icon(Icons.cancel, color: Colors.redAccent, size: 20),
                          ],
                        ),
                      ),
                    ),
                  );
                }),

                // Loading spinner while waiting for explanation
                if (selectedOption.isNotEmpty && !isAnswered) ...[
                  const SizedBox(height: 20),
                  const Center(
                    child: Column(
                      children: [
                        SizedBox(
                          width: 40,
                          height: 40,
                          child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF2196F3)),
                        ),
                        SizedBox(height: 10),
                        Text("🤖 AI is generating explanation...", style: TextStyle(color: Colors.white70, fontStyle: FontStyle.italic)),
                      ],
                    ),
                  ),
                ],

                // Explanation section
                if (isAnswered) ...[
                  const SizedBox(height: 20),
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.05),
                      borderRadius: BorderRadius.circular(15),
                      border: Border.all(color: Colors.white10),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(Icons.lightbulb_outline, color: accentColor, size: 20),
                            const SizedBox(width: 8),
                            Text("Explanation", style: TextStyle(fontWeight: FontWeight.bold, color: accentColor, fontSize: 18)),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Text(q['explanation'] ?? "Standard logic applies.", style: const TextStyle(color: Colors.white70, fontSize: 16, height: 1.5)),
                        const SizedBox(height: 20),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            if (currentIndex < questions.length - 1)
                              Expanded(
                                child: ElevatedButton(
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: Colors.white10,
                                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                                  ),
                                  onPressed: nextQuestion,
                                  child: const Text("Next →", style: TextStyle(color: Colors.white70)),
                                ),
                              ),
                            if (currentIndex < questions.length - 1)
                              const SizedBox(width: 12),
                            Expanded(
                              child: ElevatedButton(
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: currentIndex < questions.length - 1 ? Colors.white10 : accentColor,
                                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                                ),
                                onPressed: () {
                                  if (currentIndex < questions.length - 1) {
                                    _completeQuiz();
                                  } else {
                                    _completeQuiz();
                                  }
                                },
                                child: Text(
                                  currentIndex < questions.length - 1 ? "Submit Quiz" : "Finish Quiz",
                                  style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
                const SizedBox(height: 40),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _tagChip(String label, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Text(label, style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w600)),
    );
  }
}