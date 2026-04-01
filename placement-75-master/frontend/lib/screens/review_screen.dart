import 'package:flutter/material.dart';

class ReviewScreen extends StatelessWidget {
  final List<dynamic> questions;
  final List<int?> userAnswers;

  const ReviewScreen({super.key, required this.questions, required this.userAnswers});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Review & Explanations")),
      body: ListView.builder(
        itemCount: questions.length,
        itemBuilder: (context, index) {
          final q = questions[index];
          Map<String, int> mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3};
          int correct = mapping[q['answer']] ?? 0;

          return Card(
            margin: const EdgeInsets.all(10),
            child: Padding(
              padding: const EdgeInsets.all(15),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("Q${index + 1}: ${q['question']}", style: const TextStyle(fontWeight: FontWeight.bold)),
                  const SizedBox(height: 10),
                  _buildRow(q['option_a'], 0, userAnswers[index], correct),
                  _buildRow(q['option_b'], 1, userAnswers[index], correct),
                  _buildRow(q['option_c'], 2, userAnswers[index], correct),
                  _buildRow(q['option_d'], 3, userAnswers[index], correct),
                  const Divider(),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(color: Colors.blue.shade50, borderRadius: BorderRadius.circular(5)),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text("💡 Explanation:", style: TextStyle(fontWeight: FontWeight.bold, color: Colors.blue)),
                        const SizedBox(height: 4),
                        Text(q['explanation'] ?? 'Review this topic in your textbook.'),
                      ],
                    ),
                  )
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildRow(String text, int idx, int? selected, int correct) {
    Color color = Colors.transparent;
    if (idx == correct) {
      color = Colors.green.shade100;
    } else if (idx == selected) {
      color = Colors.red.shade100;
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(8),
      margin: const EdgeInsets.symmetric(vertical: 2),
      decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(4)),
      child: Text(text),
    );
  }
}