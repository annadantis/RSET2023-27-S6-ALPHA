import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class AuthProvider with ChangeNotifier {
  final String baseUrl = 'http://127.0.0.1:8000'; 
  String? username;

  Future<bool> login(String user, String pass) async {
    try {
      String cleanUser = user.trim();
      String cleanPass = pass.trim();
      
      final res = await http.post(
        Uri.parse('$baseUrl/login'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'username': cleanUser, 'password': cleanPass}),
      );

      if (res.statusCode == 200) {
        username = cleanUser;
        notifyListeners();
        return true;
      }
    } catch (e) {
      print("LOGIN ERROR: $e");
    }
    return false;
  }

  void setTeacherSession(String user) {
    username = user;
    notifyListeners();
  }

  Future<bool> register(String user, String pass, [String? branch, String? role]) async {
    try {
      String cleanUser = user.trim();
      String cleanPass = pass.trim();

      final res = await http.post(
        Uri.parse('$baseUrl/register'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'username': cleanUser, 
          'password': cleanPass,
          'branch': branch,
          'role': role ?? 'student'
        }),
      );
      
      print("REGISTRATION RESPONSE: ${res.statusCode} ${res.body}");
      return res.statusCode == 200;
    } catch (e) {
      print("REGISTER ERROR: $e");
      return false;
    }
  }

  Future<bool> updateBranch(String branch) async {
    if (username == null) {
      print("❌ UPDATE BRANCH ERROR: username is null!");
      return false;
    }
    
    print("📤 Sending update_branch request:");
    print("   Username: $username");
    print("   Branch: $branch");
    
    try {
      final res = await http.post(
        Uri.parse('$baseUrl/update_branch'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'username': username,
          'branch': branch
        }),
      );
      
      print("📥 Response status: ${res.statusCode}");
      print("📥 Response body: ${res.body}");
      
      return res.statusCode == 200;
    } catch (e) {
      print("❌ UPDATE BRANCH ERROR: $e");
      return false;
    }
  }

  void logout() {
    username = null;
    notifyListeners();
  }
}