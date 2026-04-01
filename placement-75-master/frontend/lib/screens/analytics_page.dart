import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class AnalyticsPage extends StatefulWidget {
  final String username;
  const AnalyticsPage({super.key, required this.username});

  @override
  State<AnalyticsPage> createState() => _AnalyticsPageState();
}

class _AnalyticsPageState extends State<AnalyticsPage> {
  List<dynamic> graphData = [];
  String level = "N/A";
  double avg = 0.0;
  bool hasData = false;

  @override
  void initState() {
    super.initState();
    fetch();
  }

  Future<void> fetch() async {
    try {
      final res = await http.get(Uri.parse('http://127.0.0.1:8000/weekly_report/${widget.username}'));
      if (res.statusCode == 200) {
        final data = json.decode(res.body);
        setState(() {
          hasData = data['has_data'];
          if (hasData) {
            graphData = data['graph_data'];
            level = data['knowledge_level'];
            avg = data['average_score'].toDouble();
          }
        });
      }
    } catch (e) { print(e); }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Your Progress")),
      body: !hasData 
          ? const Center(child: Text("Take some quizzes to generate data!"))
          : SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  _card("Average Score", avg.toString(), Colors.blue),
                  const SizedBox(height: 15),
                  _card("Knowledge Status", level, Colors.green),
                  const SizedBox(height: 30),
                  const Text("Performance History", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 20),
                  SizedBox(height: 300, child: _lineChart()),
                ],
              ),
            ),
    );
  }

  Widget _card(String title, String val, Color col) {
    return Card(
      child: ListTile(
        title: Text(title),
        trailing: Text(val, style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: col)),
      ),
    );
  }

  Widget _lineChart() {
    return LineChart(
      LineChartData(
        lineBarsData: [
          LineChartBarData(
            spots: graphData.asMap().entries.map((e) => FlSpot(e.key.toDouble(), e.value['score'].toDouble())).toList(),
            isCurved: true,
            color: Colors.blueAccent,
            belowBarData: BarAreaData(show: true, color: Colors.blueAccent.withOpacity(0.1)),
          )
        ],
      ),
    );
  }
}