import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiConfig {
  // Use 10.0.2.2 if using Android Emulator, or your local IP if testing on a physical device.
  // For Windows/Web, localhost works.
  static const String baseUrl = "http://localhost:8000";

  // ---------------- INTERVIEW ENDPOINTS ----------------

  static Future<Map<String, dynamic>> evaluateInterview(String filePath) async {
    var request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/evaluate_interview'),
    );
    request.files.add(await http.MultipartFile.fromPath('audio', filePath));

    var streamedResponse = await request.send();
    var response = await http.Response.fromStream(streamedResponse);
    return json.decode(response.body);
  }

  // ---------------- GD MODULE ENDPOINTS ----------------

  /// Fetches a random GD topic from the MySQL database
  static Future<Map<String, dynamic>> fetchGDTopic() async {
    try {
      // Ensure the path matches the one that worked in your browser!
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

  /// Submits audio + video to GD evaluation backend along with conversation history
  /// Submits audio + video to GD evaluation backend along with conversation history
  static Future<Map<String, dynamic>> submitGD({
    required String topicId, 
    required List<File> audioFiles,
    required File videoFile,
    required String botContext, 
  }) async {
    try {
      // Ensure the endpoint matches your router prefix (e.g., /gd_module)
      final url = Uri.parse('$baseUrl/gd_module/submit');

      var request = http.MultipartRequest('POST', url);

      // 1. Add fields (FastAPI Form data)
      request.fields['topic_id'] = topicId;
      request.fields['bot_context'] = botContext; // Crucial for Phase 2 Evaluation

      // 2. Add the Audio (MultipartFile)
      for (var file in audioFiles) {
        request.files.add(await http.MultipartFile.fromPath(
          'audio',
          file.path,
        ));
      }

      // 3. Add the Video (MultipartFile)
      request.files.add(await http.MultipartFile.fromPath(
        'video',
        videoFile.path,
      ));

      // Send the request
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        // Detailed error logging to help debug logic issues
        print("GD Backend Error: ${response.statusCode} - ${response.body}");
        throw Exception("GD Evaluation failed. Status: ${response.statusCode}");
      }
    } catch (e) {
      print("Submit Network Error: $e");
      throw Exception("Failed to connect to the evaluation server.");
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
  static Future<String> fetchNewsSummary(String title) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/news/summary'),
        headers: {"Content-Type": "application/json"},
        body: json.encode({"title": title}),
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

  // ---------------- PERFORMANCE & ANALYTICS ----------------

  static Future<List<dynamic>> fetchUserHistory(String username) async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/history/$username'));
      if (response.statusCode == 200) {
        return json.decode(response.body);
      }
      return [];
    } catch (e) {
      print("History Fetch Error: $e");
      return [];
    }
  }

  static Future<Map<String, dynamic>> fetchPerformanceByDate(String username, String date) async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/performance_by_date/$username/$date'));
      if (response.statusCode == 200) {
        return json.decode(response.body);
      }
      throw Exception("Failed to fetch performance for $date");
    } catch (e) {
      print("Date Performance error: $e");
      rethrow;
    }
  }

  static Future<Map<String, dynamic>> fetchSessionDetail(String category, int sessionId) async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/session_detail/${category.toUpperCase()}/$sessionId'));
      if (response.statusCode == 200) {
        return json.decode(response.body);
      }
      throw Exception("Failed to load session details");
    } catch (e) {
      print("Session Detail error: $e");
      rethrow;
    }
  }
}