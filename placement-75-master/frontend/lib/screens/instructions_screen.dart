import 'package:flutter/material.dart';

/// A screen that shows test guidelines to the user before they begin a quiz or session.
/// On acceptance, it calls [onStart] to navigate to the actual screen.
class InstructionsScreen extends StatefulWidget {
  final String category; // "APTITUDE", "TECHNICAL", "GD", "INTERVIEW"
  final VoidCallback onStart;

  const InstructionsScreen({
    super.key,
    required this.category,
    required this.onStart,
  });

  @override
  State<InstructionsScreen> createState() => _InstructionsScreenState();
}

class _InstructionsScreenState extends State<InstructionsScreen>
    with TickerProviderStateMixin {
  late AnimationController _fadeController;
  late Animation<double> _fadeAnim;
  bool _agreed = false;

  @override
  void initState() {
    super.initState();
    _fadeController = AnimationController(vsync: this, duration: const Duration(milliseconds: 500));
    _fadeAnim = CurvedAnimation(parent: _fadeController, curve: Curves.easeIn);
    _fadeController.forward();
  }

  @override
  void dispose() {
    _fadeController.dispose();
    super.dispose();
  }

  _CategoryInfo get _info {
    switch (widget.category.toUpperCase()) {
      case 'APTITUDE':
        return _CategoryInfo(
          title: 'Aptitude Test',
          icon: Icons.psychology_outlined,
          color: Colors.orangeAccent,
          description: 'This test evaluates your quantitative reasoning, logical thinking, and problem-solving skills.',
          rules: [
            '10 multiple-choice questions.',
            'Total time: 5 minutes (30 seconds/question).',
            'No negative marking.',
            'Once an answer is submitted it cannot be changed.',
            'You can mark questions for review and revisit them.',
            'The test will auto-submit when the timer runs out.',
            'Each question is graded immediately and tracked for analytics.',
          ],
        );
      case 'TECHNICAL':
        return _CategoryInfo(
          title: 'Technical Test',
          icon: Icons.code,
          color: Colors.blueAccent,
          description: 'This test assesses your technical knowledge in your engineering branch.',
          rules: [
            '10 multiple-choice questions from your branch/field.',
            'Total time: 5 minutes.',
            'No negative marking.',
            'Once an answer is submitted it cannot be changed.',
            'You can mark questions for review and revisit them.',
            'The test will auto-submit when the timer runs out.',
            'Performance is used to adjust your difficulty level over time.',
          ],
        );
      case 'GD':
        return _CategoryInfo(
          title: 'Group Discussion',
          icon: Icons.groups_outlined,
          color: Colors.cyanAccent,
          description: 'A collaborative AI-moderated discussion on a trending topic.',
          rules: [
            'Raise your hand (pan tool icon) to request time to speak.',
            'You must speak your answer clearly',
            'Maintain eye contact with the camera for a better score.',
            'Your voice and camera are recorded for comprehensive evaluation.',
            'Ensure you are in a well-lit, quiet environment.',
            'Your audio is transcribed and evaluated by AI.',
            'Scoring is based on content accuracy and camera input.',
            'Leaving the session mid-way will discard your result.',

          ],
        );
      case 'INTERVIEW':
      default:
        return _CategoryInfo(
          title: 'Mock Interview',
          icon: Icons.mic_none,
          color: Colors.pinkAccent,
          description: 'A simulated technical/behavioral interview. An AI interviewer will ask questions and evaluate your spoken responses.',
          rules: [
            'Questions are based on your engineering branch.',
            'You must speak your answer clearly within the given time.',
            'Your audio is transcribed and evaluated by AI.',
            'Scoring is based on content accuracy and camera input.',
            'Ensure you are in a well-lit, quiet environment.',
            'Your camera will be active for  analysis.',
            'Leaving the session mid-way will discard your result.',
          ],
        );
    }
  }

  @override
  Widget build(BuildContext context) {
    final info = _info;
    return Scaffold(
      backgroundColor: const Color(0xFF0F0C29),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.close, color: Colors.white70),
          onPressed: () => Navigator.pop(context),
        ),
        title: Row(
          children: [
            Icon(info.icon, color: info.color, size: 20),
            const SizedBox(width: 8),
            Text('Guidelines', style: TextStyle(color: info.color, fontWeight: FontWeight.bold)),
          ],
        ),
      ),
      body: FadeTransition(
        opacity: _fadeAnim,
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header Card
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [info.color.withOpacity(0.15), info.color.withOpacity(0.03)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: info.color.withOpacity(0.3)),
                ),
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: info.color.withOpacity(0.15),
                        shape: BoxShape.circle,
                      ),
                      child: Icon(info.icon, color: info.color, size: 36),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(info.title, style: TextStyle(color: info.color, fontWeight: FontWeight.w900, fontSize: 22)),
                          const SizedBox(height: 6),
                          Text(info.description, style: const TextStyle(color: Colors.white60, fontSize: 13, height: 1.5)),
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 28),

              // Rules Section
              _sectionHeader('Rules & Format', Icons.rule, Colors.white),
              const SizedBox(height: 12),
              ...info.rules.asMap().entries.map((e) => _ruleItem(e.key + 1, e.value, info.color)),

              const SizedBox(height: 24),

              // Warning Note
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.orange.withOpacity(0.08),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: Colors.orangeAccent.withOpacity(0.4)),
                ),
                child: const Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.warning_amber_rounded, color: Colors.orangeAccent, size: 20),
                    SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Your progress will NOT be saved if you exit or close the session mid-way. The test result will be discarded and not recorded in your analytics.',
                        style: TextStyle(color: Colors.orangeAccent, fontSize: 13, height: 1.5),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 24),

              // Agree Checkbox
              InkWell(
                onTap: () => setState(() => _agreed = !_agreed),
                borderRadius: BorderRadius.circular(10),
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  child: Row(
                    children: [
                      AnimatedContainer(
                        duration: const Duration(milliseconds: 200),
                        width: 24,
                        height: 24,
                        decoration: BoxDecoration(
                          color: _agreed ? info.color : Colors.transparent,
                          borderRadius: BorderRadius.circular(6),
                          border: Border.all(color: _agreed ? info.color : Colors.white30, width: 2),
                        ),
                        child: _agreed
                            ? const Icon(Icons.check, color: Colors.black, size: 16)
                            : null,
                      ),
                      const SizedBox(width: 12),
                      const Expanded(
                        child: Text(
                          'I have read and understood the instructions above.',
                          style: TextStyle(color: Colors.white70, fontSize: 14),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 20),

              // Start Button
              SizedBox(
                width: double.infinity,
                height: 54,
                child: ElevatedButton(
                  onPressed: _agreed
                      ? () {
                          Navigator.pop(context);
                          widget.onStart();
                        }
                      : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: info.color,
                    disabledBackgroundColor: Colors.white12,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                    elevation: _agreed ? 8 : 0,
                    shadowColor: info.color.withOpacity(0.5),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.play_arrow, color: _agreed ? Colors.white : Colors.white30),
                      const SizedBox(width: 8),
                      Text(
                        'Start ${info.title}',
                        style: TextStyle(
                          color: _agreed ? Colors.white : Colors.white30,
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 32),
            ],
          ),
        ),
      ),
    );
  }

  Widget _sectionHeader(String title, IconData icon, Color color) {
    return Row(
      children: [
        Icon(icon, color: color, size: 18),
        const SizedBox(width: 8),
        Text(title, style: TextStyle(color: color, fontWeight: FontWeight.bold, fontSize: 16)),
      ],
    );
  }

  Widget _ruleItem(int num, String text, Color accentColor) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 24,
            height: 24,
            decoration: BoxDecoration(
              color: accentColor.withOpacity(0.15),
              shape: BoxShape.circle,
              border: Border.all(color: accentColor.withOpacity(0.4)),
            ),
            child: Center(
              child: Text('$num', style: TextStyle(color: accentColor, fontSize: 11, fontWeight: FontWeight.bold)),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(text, style: const TextStyle(color: Colors.white70, fontSize: 14, height: 1.45)),
          ),
        ],
      ),
    );
  }
}

class _CategoryInfo {
  final String title;
  final IconData icon;
  final Color color;
  final String description;
  final List<String> rules;

  const _CategoryInfo({
    required this.title,
    required this.icon,
    required this.color,
    required this.description,
    required this.rules,
  });
}
