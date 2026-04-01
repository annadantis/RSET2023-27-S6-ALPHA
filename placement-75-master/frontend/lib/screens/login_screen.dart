import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../providers/auth_provider.dart';
import 'dashboard_screen.dart';
import 'register_screen.dart';
import 'branch_selection_screen.dart';
import 'teacher_dashboard_screen.dart';

import 'forgot_password_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _userController = TextEditingController();
  final _passController = TextEditingController();
  String _selectedRole = "Student"; // To track role selection
  bool _obscurePassword = true;

  @override
  Widget build(BuildContext context) {
    // Deep dark theme colors from your figure
    const Color scaffoldBg = Color(0xFF0F0C29);
    
    const Color accentColor = Color(0xFF2196F3); // Premium Blue Accent

    return Scaffold(
      backgroundColor: scaffoldBg,
      body: Row(
        children: [
          // --- LEFT SIDE: Brand Info ---
          Expanded(
            flex: 1,
            child: Container(
              padding: const EdgeInsets.all(40),
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [Color(0xFF1A237E), Color(0xFF4A148C)],
                ),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.school, size: 60, color: Colors.white),
                  const SizedBox(height: 20),
                  const Text(
                    "AceAIre - AI Placement Assistant",
                    style: TextStyle(fontSize: 42, fontWeight: FontWeight.bold, color: Colors.white),
                  ),
                  const Text(
                    "Comprehensive preparation platform for campus placements",
                    style: TextStyle(fontSize: 18, color: Colors.white70),
                  ),
                  const SizedBox(height: 40),
                  _buildFeatureRow(Icons.check_circle_outline, "Smart Analytics", "Track performance with detailed insights"),
                  _buildFeatureRow(Icons.mic, "AI-Powered Practice", "Interview and GD simulation with real-time feedback"),
                  _buildFeatureRow(Icons.library_books, "Comprehensive Coverage", "Technical, aptitude, and soft skills training"),
                ],
              ),
            ),
          ),

          // --- RIGHT SIDE: Login Form ---
          Expanded(
            flex: 1,
            child: Center(
              child: Container(
                constraints: const BoxConstraints(maxWidth: 450),
                padding: const EdgeInsets.all(40),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text("Welcome Back", style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: Colors.white)),
                    const Text("Sign in to continue your preparation", style: TextStyle(color: Colors.white54)),
                    const SizedBox(height: 30),

                    // Role Selection (Student/Teacher)
                    const Text("Select Role", style: TextStyle(color: Colors.white70, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 10),
                    Row(
                      children: [
                        _roleButton("Student", Icons.person_outline, _selectedRole == "Student"),
                        const SizedBox(width: 15),
                        _roleButton("Teacher", Icons.groups_outlined, _selectedRole == "Teacher"),
                      ],
                    ),
                    const SizedBox(height: 25),

                    // Input Fields
                    _buildTextField(_selectedRole == "Student" ? "College Email (@rajagiri.edu.in)/ Username" : "Username / Email", _userController, _selectedRole == "Student" ? Icons.email_outlined : Icons.account_circle_outlined),
                    const SizedBox(height: 20),
                    _buildTextField("Password", _passController, Icons.lock_outline, isPassword: true),

                    const SizedBox(height: 15),

                    const SizedBox(height: 20),
                    // Login button with role-based authentication
                    ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: accentColor,
                        minimumSize: const Size(double.infinity, 55),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      ),
                      onPressed: () async {
                        // Teacher login logic
                        if (_selectedRole == "Teacher") {
                          try {
                            final response = await http.post(
                              Uri.parse('http://127.0.0.1:8000/teacher/login'),
                              headers: {'Content-Type': 'application/json'},
                              body: jsonEncode({
                                'username': _userController.text.trim(),
                                'password': _passController.text,
                              }),
                            );

                            if (response.statusCode == 200 && context.mounted) {
                              final data = jsonDecode(response.body);
                              Navigator.pushReplacement(
                                context,
                                MaterialPageRoute(
                                  builder: (_) => TeacherDashboardScreen(
                                    teacherName: data['username'],
                                  ),
                                ),
                              );
                            } else {
                              if (context.mounted) {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(content: Text("Teacher Login Failed - Invalid Credentials")),
                                );
                              }
                            }
                          } catch (e) {
                            if (context.mounted) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(content: Text("Login Error: $e")),
                              );
                            }
                          }
                          return;
                        }
                        
                        // Student login logic (existing code)
                        final authProvider = Provider.of<AuthProvider>(context, listen: false);
                        final success = await authProvider.login(_userController.text, _passController.text);

                        if (success && context.mounted) {
                          // Fetch user dashboard data to check branch
                          try {
                            final response = await http.get(
                              Uri.parse('${authProvider.baseUrl}/dashboard/${authProvider.username}'),
                            );
                            
                            if (response.statusCode == 200) {
                              final data = jsonDecode(response.body);
                              final branch = data['branch'];
                              
                              if (branch == null || branch == '') {
                                // No branch set - redirect to branch selection
                                Navigator.pushReplacement(
                                  context,
                                  MaterialPageRoute(builder: (_) => const BranchSelectionScreen()),
                                );
                              } else {
                                // Branch exists - go to dashboard
                                Navigator.pushReplacement(
                                  context,
                                  MaterialPageRoute(builder: (_) => const DashboardScreen()),
                                );
                              }
                            } else {
                              // Fallback to dashboard if API fails
                              Navigator.pushReplacement(
                                context,
                                MaterialPageRoute(builder: (_) => const DashboardScreen()),
                              );
                            }
                          } catch (e) {
                            print('Error checking branch: $e');
                            // Fallback to dashboard
                            Navigator.pushReplacement(
                              context,
                              MaterialPageRoute(builder: (_) => const DashboardScreen()),
                            );
                          }
                        } else {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(content: Text("Login Failed")),
                          );
                        }
                      },
                      child: Text("Sign In as $_selectedRole", style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white)),
                    ),

                    const SizedBox(height: 20),
                    Center(
                      child: TextButton(
                        onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const ForgotPasswordScreen())),
                        child: const Text("Forgot Password?", style: TextStyle(color: Colors.white70)),
                      ),
                    ),

                    Center(
                      child: TextButton(
                        onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const RegisterScreen())),
                        child: const Text("New User? Register here", style: TextStyle(color: Colors.white70)),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // --- UI HELPER COMPONENTS ---

  Widget _roleButton(String title, IconData icon, bool isSelected) {
    return Expanded(
      child: GestureDetector(
        onTap: () => setState(() => _selectedRole = title),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 15),
          decoration: BoxDecoration(
            color: isSelected ? const Color.fromARGB(255, 34, 27, 166).withOpacity(0.2) : Colors.white.withOpacity(0.05),
            border: Border.all(color: isSelected ? const Color.fromARGB(255, 41, 34, 163) : Colors.white12, width: 2),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            children: [
              Icon(icon, color: isSelected ? Colors.white : Colors.white54),
              const SizedBox(height: 5),
              Text(title, style: TextStyle(color: isSelected ? Colors.white : Colors.white54)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTextField(String label, TextEditingController controller, IconData icon, {bool isPassword = false}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(color: Colors.white70, fontSize: 14)),
        const SizedBox(height: 8),
        TextField(
          controller: controller,
          obscureText: isPassword && _obscurePassword,
          style: const TextStyle(color: Colors.white),
          decoration: InputDecoration(
            prefixIcon: Icon(icon, color: Colors.white38),
            suffixIcon: isPassword 
              ? IconButton(
                  icon: Icon(_obscurePassword ? Icons.visibility_off : Icons.visibility, color: Colors.white38),
                  onPressed: () => setState(() => _obscurePassword = !_obscurePassword),
                )
              : null,
            filled: true,
            fillColor: Colors.white.withOpacity(0.05),
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
            hintText: "Enter your ${label.toLowerCase()}",
            hintStyle: const TextStyle(color: Colors.white24),
          ),
        ),
      ],
    );
  }

  Widget _buildFeatureRow(IconData icon, String title, String subtitle) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 25.0),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(color: Colors.white10, borderRadius: BorderRadius.circular(10)),
            child: Icon(icon, color: Colors.white, size: 24),
          ),
          const SizedBox(width: 20),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)),
              Text(subtitle, style: const TextStyle(color: Colors.white54, fontSize: 13)),
            ],
          )
        ],
      ),
    );
  }

  @override
  void dispose() {
    _userController.dispose();
    _passController.dispose();
    super.dispose();
  }
}