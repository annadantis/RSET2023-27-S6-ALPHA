import 'dart:convert';
import 'package:http/http.dart' as http;

class TeacherApiService {
  static const String baseUrl = 'http://127.0.0.1:8000';

  // Teacher Login
  static Future<Map<String, dynamic>> login(String username, String password) async {
    final response = await http.post(
      Uri.parse('$baseUrl/teacher/login'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'username': username,
        'password': password,
      }),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Login failed: ${response.body}');
    }
  }

  // Teacher Registration
  static Future<Map<String, dynamic>> register(String username, String email, String password, String secretCode) async {
    final response = await http.post(
      Uri.parse('$baseUrl/teacher/register'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'username': username,
        'email': email,
        'password': password,
        'secret_code': secretCode,
      }),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Registration failed: ${response.body}');
    }
  }

  // Get All Students
  static Future<Map<String, dynamic>> getStudents({String? branch, String? search}) async {
    String url = '$baseUrl/teacher/students';
    List<String> queryParams = [];
    
    if (branch != null && branch.isNotEmpty && branch != 'All') {
      queryParams.add('branch=$branch');
    }
    if (search != null && search.isNotEmpty) {
      queryParams.add('search=$search');
    }
    
    if (queryParams.isNotEmpty) {
        url += '?${queryParams.join('&')}';
    }

    final response = await http.get(Uri.parse(url));

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load students');
    }
  }

  // Get Student Progress
  static Future<Map<String, dynamic>> getStudentProgress(String username) async {
    final response = await http.get(
      Uri.parse('$baseUrl/teacher/students/$username/progress'),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load student progress');
    }
  }

  // Get Dashboard Overview
  static Future<Map<String, dynamic>> getDashboardOverview() async {
    final response = await http.get(
      Uri.parse('$baseUrl/teacher/dashboard/overview'),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load dashboard overview');
    }
  }

  // Get Branch Analytics
  static Future<Map<String, dynamic>> getBranchAnalytics(String branch) async {
    final response = await http.get(
      Uri.parse('$baseUrl/teacher/dashboard/branch/$branch'),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load branch analytics');
    }
  }

  // Get Branch Ranking Leaderboard
  static Future<Map<String, dynamic>> getBranchRanking(String branch) async {
    final response = await http.get(
      Uri.parse('$baseUrl/teacher/dashboard/branch/$branch/ranking'),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load branch ranking');
    }
  }

  // Get Daily Activity
  static Future<Map<String, dynamic>> getDailyActivity({String? date}) async {
    String url = '$baseUrl/teacher/dashboard/activity';
    if (date != null && date.isNotEmpty) {
      url += '?date_str=$date';
    }

    final response = await http.get(Uri.parse(url));

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load daily activity');
    }
  }

  // Get Batch Trends (New)
  static Future<Map<String, dynamic>> getBatchTrends() async {
    final response = await http.get(
      Uri.parse('$baseUrl/teacher/dashboard/batch_trends'),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load batch trends');
    }
  }

  // Get AI Recommendations (New)
  static Future<Map<String, dynamic>> getAiRecommendations() async {
    final response = await http.get(
      Uri.parse('$baseUrl/teacher/dashboard/ai_recommendations'),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load AI recommendations');
    }
  }

  // Get Interview Session Detail (New)
  static Future<Map<String, dynamic>> getInterviewSessionDetail(int sessionId) async {
    final response = await http.get(
      Uri.parse('$baseUrl/teacher/interviews/$sessionId'),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load interview session detail');
    }
  }

  // Get Live Activity Pulse (New)
  static Future<Map<String, dynamic>> getLivePulse() async {
    final response = await http.get(
      Uri.parse('$baseUrl/teacher/dashboard/live_pulse'),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load live pulse activity');
    }
  }

  // Send Suggestion to Student (New)
  static Future<Map<String, dynamic>> sendSuggestion(String studentUsername, String teacherUsername, String message) async {
    print("DEBUG: Sending suggestion from $teacherUsername to $studentUsername");
    final response = await http.post(
      Uri.parse('$baseUrl/teacher/students/$studentUsername/suggest'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'teacher_username': teacherUsername,
        'message': message,
      }),
    );

    print("DEBUG: Suggestion response status: ${response.statusCode}");
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to send suggestion: ${response.body}');
    }
  }

  // Get Suggestions sent to a specific student (New)
  static Future<List<dynamic>> getSuggestions(String studentUsername) async {
    final response = await http.get(
      Uri.parse('$baseUrl/suggestions/$studentUsername'),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['suggestions'] ?? [];
    } else {
      throw Exception('Failed to fetch suggestions: ${response.body}');
    }
  }
}

