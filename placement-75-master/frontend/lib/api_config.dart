import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiConfig {
  // Use 10.0.2.2 if using Android Emulator, or your local IP if testing on a physical device.
  // For Windows/Web, localhost works.
  static const String baseUrl = "http://127.0.0.1:8000";

  // ---------------- INTERVIEW ENDPOINTS ----------------

  static Future<Map<String, dynamic>> evaluateInterview(String filePath, {String username = "Anonymous"}) async {
    var request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/evaluate_interview'),
    );
    request.fields['username'] = username;
    request.files.add(await http.MultipartFile.fromPath('audio', filePath));

    var streamedResponse = await request.send();
    var response = await http.Response.fromStream(streamedResponse);
    return json.decode(response.body);
  }

  // ---------------- GD MODULE ENDPOINTS ----------------

  /// Fetches a random GD topic from the MySQL database
  static Future<Map<String, dynamic>> fetchGDTopic() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/gd_module/gd/topic'));

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception("Server returned ${response.statusCode}");
      }
    } catch (e) {
      print("Fetch Error: $e");
      throw Exception("Failed to load GD topic");
    }
  }

  /// Submits audio + video to GD evaluation backend
  static Future<Map<String, dynamic>> submitGD({
    required String topicId,
    required List<File> audioFiles,
    required File videoFile,
    required String botContext,
    String username = "Anonymous",
  }) async {
    try {
      final url = Uri.parse('$baseUrl/gd_module/submit');
      var request = http.MultipartRequest('POST', url);

      request.fields['topic_id'] = topicId;
      request.fields['username'] = username;
      request.fields['bot_context'] = botContext;

      // Add multiple audio files
      for (var file in audioFiles) {
        request.files.add(await http.MultipartFile.fromPath('audio', file.path));
      }
      
      request.files.add(await http.MultipartFile.fromPath('video', videoFile.path));

      final streamedResponse = await request.send().timeout(const Duration(minutes: 5));
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        print("Server Error: ${response.body}");
        throw Exception("GD Evaluation failed: ${response.body}");
      }
    } catch (e) {
      print("Submit Error: $e");
      throw Exception("Network error during submission");
    }
  }

  /// Simulates a multi-bot GD for a given topic
  static Future<Map<String, dynamic>> simulateGD(int topicId) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/gd_module/simulate'),
        headers: {"Content-Type": "application/json"},
        body: json.encode({"topic_id": topicId}),
      );

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception("Simulation failed: ${response.statusCode}");
      }
    } catch (e) {
      print("Simulation Error: $e");
      throw Exception("Failed to start GD simulation");
    }
  }

  // ---------------- NEWS & TRENDS ENDPOINTS ----------------

  /// Fetches latest industry news from Hacker News proxy
  static Future<List<dynamic>> fetchLatestNews() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/news/latest'));
      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception("Failed to load news: ${response.statusCode}");
      }
    } catch (e) {
      print("News Fetch Error: $e");
      return [];
    }
  }

  /// Fetches a 3-sentence AI briefing of current trends
  static Future<String> fetchIndustryTrendsBriefing() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/news/trends-briefing'));
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return data['briefing'] ?? "No briefing available.";
      } else {
        throw Exception("Failed to load briefing: ${response.statusCode}");
      }
    } catch (e) {
      print("Briefing Fetch Error: $e");
      return "Current industry trends are focused on AI and system efficiency.";
    }
  }

  /// Fetches an AI-generated 1-sentence summary for a specific news title
  static Future<String> fetchNewsSummary(String title, {String? url}) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/news/summary'),
        headers: {"Content-Type": "application/json"},
        body: json.encode({
          "title": title,
          "url": url,
        }),
      );
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return data['summary'] ?? title;
      } else {
        return title;
      }
    } catch (e) {
      print("Summary Fetch Error: $e");
      return title;
    }
  }

  /// Stops any ongoing news summary speech
  static Future<void> stopSpeech() async {
    try {
      await http.post(Uri.parse('$baseUrl/news/stop-speech'));
    } catch (e) {
      print("Stop Speech Error: $e");
    }
  }

  static Future<List<dynamic>> fetchSuggestions(String username) async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/suggestions/$username'));
      if (response.statusCode == 200) {
        return json.decode(response.body)['suggestions'] ?? [];
      } else {
        throw Exception("Failed to load suggestions: ${response.statusCode}");
      }
    } catch (e) {
      print("Suggestions Fetch Error: $e");
      return [];
    }
  }

  // ---------------- HISTORY & REPORTS ----------------

  static Future<List<dynamic>> fetchUserHistory(String username) async {
    final response = await http.get(Uri.parse('$baseUrl/history/$username'));
    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception("Failed to load history");
    }
  }

  static Future<Map<String, dynamic>> fetchSessionDetail(String category, int sessionId) async {
    final response = await http.get(Uri.parse('$baseUrl/session_detail/$category/$sessionId'));
    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception("Failed to load session details");
    }
  }

  static Future<Map<String, dynamic>> fetchPerformanceByDate(String username, String dateStr) async {
    final response = await http.get(Uri.parse('$baseUrl/performance_by_date/$username/$dateStr'));
    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception("Failed to load performance by date");
    }
  }

  static Future<void> markSuggestionRead(int id) async {
    try {
      final response = await http.post(Uri.parse('$baseUrl/suggestions/$id/read'));
      if (response.statusCode != 200) {
        print("Failed to mark suggestion read: ${response.body}");
      }
    } catch (e) {
      print("Mark Suggestion Read Error: $e");
    }
  }
}