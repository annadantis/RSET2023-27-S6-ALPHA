import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../providers/auth_provider.dart';
import '../api_config.dart';
import 'login_screen.dart';
import 'otp_screen.dart';

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final _userController = TextEditingController();
  final _emailController = TextEditingController(); 
  final _passController = TextEditingController();
  final _secretCodeController = TextEditingController(); 
  String _selectedRole = "Student"; 
  String? _selectedBranch; 
  bool _obscurePassword = true;
  bool _isLoading = false;

  @override
  Widget build(BuildContext context) {
    const Color scaffoldBg = Color(0xFF0F0C29);
    const Color accentColor = Color(0xFF2196F3);

    return Scaffold(
      backgroundColor: scaffoldBg,
      body: Row(
        children: [
          // --- LEFT SIDE: Brand & Registration Info ---
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
                  const Icon(Icons.person_add_outlined, size: 60, color: Colors.white),
                  const SizedBox(height: 20),
                  const Text(
                    "Join the Future of Placement",
                    style: TextStyle(fontSize: 42, fontWeight: FontWeight.bold, color: Colors.white),
                  ),
                  const Text(
                    "Create an account to access AI-powered mock interviews and personalized analytics.",
                    style: TextStyle(fontSize: 18, color: Colors.white70),
                  ),
                  const SizedBox(height: 40),
                  _buildFeatureRow(Icons.verified_user_outlined, "Secure Profile", "Your data is encrypted and used only for your growth."),
                  _buildFeatureRow(Icons.auto_graph, "Personalized Roadmap", "Get custom preparation paths based on your performance."),
                  _buildFeatureRow(Icons.bolt, "Instant Access", "Start practicing immediately after signing up."),
                ],
              ),
            ),
          ),

          // --- RIGHT SIDE: Registration Form ---
          Expanded(
            flex: 1,
            child: Center(
              child: Container(
                constraints: const BoxConstraints(maxWidth: 450),
                child: ListView(
                  shrinkWrap: true,
                  padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 40),
                  children: [
                    const Text("Create Account", style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: Colors.white)),
                    const Text("Fill in the details to get started", style: TextStyle(color: Colors.white54)),
                    const SizedBox(height: 30),
  
                    // Role Selection
                    const Text("I am a:", style: TextStyle(color: Colors.white70, fontWeight: FontWeight.bold)),
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
                    if (_selectedRole == "Student") ...[
                      _buildTextField("College Email (@rajagiri.edu.in)", _emailController, Icons.email_outlined),
                    ] else ...[
                      _buildTextField("College Email (@rajagiritech.edu.in)", _emailController, Icons.email_outlined),
                    ],
                    const SizedBox(height: 20),
                    
                    _buildTextField("Full Name (Auto Caps)", _userController, Icons.account_circle_outlined, textCapitalization: TextCapitalization.characters),
                    const SizedBox(height: 20),

                    _buildTextField("Create Password", _passController, Icons.lock_reset_outlined, isPassword: true),
                    const SizedBox(height: 20),
                    
                    if (_selectedRole == "Teacher") ...[
                      _buildTextField("Secret Access Code", _secretCodeController, Icons.vpn_key_outlined, isPassword: true),
                      const SizedBox(height: 20),
                    ],
  
                    // Branch Selection (Only if Student)
                    if (_selectedRole == "Student") ...[
                      const Text("Select Branch:", style: TextStyle(color: Colors.white70, fontSize: 14)),
                      const SizedBox(height: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.05),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: DropdownButtonHideUnderline(
                          child: DropdownButton<String>(
                            value: _selectedBranch,
                            hint: const Text("Choose Branch", style: TextStyle(color: Colors.white38)),
                            isExpanded: true,
                            dropdownColor: const Color(0xFF161625),
                            icon: const Icon(Icons.arrow_drop_down, color: Colors.white54),
                            style: const TextStyle(color: Colors.white),
                            items: ["CSE", "IT", "AI&DS", "CSBS", "ECE", "EEE", "AEI", "MECH", "CIVIL"].map((String branch) {
                              return DropdownMenuItem<String>(
                                value: branch,
                                child: Text(branch),
                              );
                            }).toList(),
                            onChanged: (val) => setState(() => _selectedBranch = val),
                          ),
                        ),
                      ),
                      const SizedBox(height: 20),
                    ],
  
                    // Registration Logic
                    _isLoading
                      ? const Center(child: CircularProgressIndicator(color: accentColor))
                      : ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: accentColor,
                        minimumSize: const Size(double.infinity, 55),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      ),
                      onPressed: () async {
                        if (_userController.text.isEmpty || _passController.text.isEmpty || _emailController.text.isEmpty) {
                           ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Please fill all required fields")));
                           return;
                        }
                        
                        // Strict domain checking based on role
                        if (_selectedRole == "Student") {
                            final emailRegExp = RegExp(r'^u\d{7}@rajagiri\.edu\.in$', caseSensitive: false);
                            if (!emailRegExp.hasMatch(_emailController.text.trim())) {
                                ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Student emails must be in the format u*******@rajagiri.edu.in")));
                                return;
                            }
                        } else if (_selectedRole == "Teacher" && !_emailController.text.toLowerCase().endsWith('@rajagiritech.edu.in')) {
                            ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Teacher emails must end in @rajagiritech.edu.in")));
                            return;
                        }

                        // Basic Password Criteria: Minimum 8 chars, 1 uppercase, 1 lowercase, 1 number, 1 special character
                        final password = _passController.text;
                        final RegExp passwordRegExp = RegExp(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$');
                        
                        if (!passwordRegExp.hasMatch(password)) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(content: Text("Password must be at least 8 characters long, include an uppercase letter, a lowercase letter, a number, and a special character (@\$!%*?&)")),
                            );
                            return;
                        }

                        setState(() => _isLoading = true);

                        if (_selectedRole == "Student") {
                          if (_selectedBranch == null) {
                             setState(() => _isLoading = false);
                             ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Please select a branch")));
                             return;
                          }
                          
                          try {
                            final response = await http.post(
                              Uri.parse('${ApiConfig.baseUrl}/auth/send-otp'),
                              headers: {'Content-Type': 'application/json'},
                              body: jsonEncode({
                                'email': _emailController.text.trim(),
                                'username': _userController.text.trim().toUpperCase(),
                                'password': _passController.text,
                                'branch': _selectedBranch,
                                'role': 'student',
                              }),
                            );
                            
                            final body = jsonDecode(response.body);
                            
                            if (response.statusCode == 200 && mounted) {
                              ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("OTP Sent! Check your email.")));
                              Navigator.push(context, MaterialPageRoute(builder: (_) => OtpScreen(email: _emailController.text.trim())));
                            } else if (mounted) {
                              ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Failed: ${body['detail']}")));
                            }
                          } catch (e) {
                             if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error: $e")));
                          }

                        } else if (_selectedRole == "Teacher") {
                          if (_secretCodeController.text.isEmpty) {
                             setState(() => _isLoading = false);
                             ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Please enter Secret Access Code")));
                             return;
                          }
                          
                          try {
                            final response = await http.post(
                              Uri.parse('${ApiConfig.baseUrl}/teacher/register'),
                              headers: {'Content-Type': 'application/json'},
                              body: jsonEncode({
                                'username': _userController.text.trim().toUpperCase(),
                                'email': _emailController.text.trim(),
                                'password': _passController.text,
                                'secret_code': _secretCodeController.text.trim(),
                              }),
                            );
                            
                            if (response.statusCode == 200 && mounted) {
                              ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Teacher Registration Successful! Please Login.")));
                              Navigator.pop(context);
                            } else if (mounted) {
                              final body = jsonDecode(response.body);
                              ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Registration Failed: ${body['detail']}")));
                            }
                          } catch (e) {
                             if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Error: $e")));
                          }
                        }
                        
                        if (mounted) setState(() => _isLoading = false);
                      },
                      child: const Text("Create Account / Send OTP", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white)),
                    ),
  
                    const SizedBox(height: 20),
                    Center(
                      child: TextButton(
                        onPressed: () => Navigator.pop(context),
                        child: const Text("Already have an account? Login here", style: TextStyle(color: Colors.white70)),
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

  Widget _roleButton(String title, IconData icon, bool isSelected) {
    return Expanded(
      child: GestureDetector(
        onTap: () => setState(() => _selectedRole = title),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 15),
          decoration: BoxDecoration(
            color: isSelected ? const Color(0xFF2196F3).withOpacity(0.2) : Colors.white.withOpacity(0.05),
            border: Border.all(color: isSelected ? const Color(0xFF2196F3) : Colors.white12, width: 2),
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

  Widget _buildTextField(String label, TextEditingController controller, IconData icon, {bool isPassword = false, TextCapitalization textCapitalization = TextCapitalization.none}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(color: Colors.white70, fontSize: 14)),
        const SizedBox(height: 8),
        TextField(
          controller: controller,
          obscureText: isPassword && _obscurePassword,
          textCapitalization: textCapitalization,
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
            hintText: label,
            hintStyle: const TextStyle(color: Colors.white24, fontSize: 14),
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
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)),
                Text(subtitle, style: const TextStyle(color: Colors.white54, fontSize: 13), softWrap: true),
              ],
            ),
          )
        ],
      ),
    );
  }

  @override
  void dispose() {
    _userController.dispose();
    _emailController.dispose();
    _passController.dispose();
    _secretCodeController.dispose();
    super.dispose();
  }
}