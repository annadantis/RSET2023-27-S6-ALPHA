import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../api_config.dart';
import 'reset_password_screen.dart';

class ForgotPasswordScreen extends StatefulWidget {
  const ForgotPasswordScreen({super.key});

  @override
  State<ForgotPasswordScreen> createState() => _ForgotPasswordScreenState();
}

class _ForgotPasswordScreenState extends State<ForgotPasswordScreen> {
  final _emailController = TextEditingController();
  final _nameController = TextEditingController();
  final _secretCodeController = TextEditingController();
  String _selectedRole = "Student";
  String? _selectedBranch;
  bool _isLoading = false;

  @override
  Widget build(BuildContext context) {
    const Color scaffoldBg = Color(0xFF0F0C29);
    const Color accentColor = Color(0xFF2196F3);

    return Scaffold(
      backgroundColor: scaffoldBg,
      body: Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 450),
          padding: const EdgeInsets.all(40),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.05),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Colors.white10),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.lock_reset, size: 60, color: Colors.white),
              const SizedBox(height: 20),
              const Text("Reset Password", style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: Colors.white)),
              const SizedBox(height: 10),
              const Text("Enter your exact registration details to receive a 6-digit confirmation code.",
                  textAlign: TextAlign.center, style: TextStyle(color: Colors.white54, fontSize: 14)),
              const SizedBox(height: 30),

              // Role Selection
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
              const SizedBox(height: 15),
              
              _buildTextField("Full Name (Exact Match)", _nameController, Icons.account_circle_outlined, textCapitalization: TextCapitalization.characters),
              const SizedBox(height: 15),

              if (_selectedRole == "Student") ...[
                const Align(alignment: Alignment.centerLeft, child: Text("Registered Branch:", style: TextStyle(color: Colors.white70, fontSize: 13))),
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12),
                  decoration: BoxDecoration(color: Colors.white.withOpacity(0.05), borderRadius: BorderRadius.circular(12)),
                  child: DropdownButtonHideUnderline(
                    child: DropdownButton<String>(
                      value: _selectedBranch,
                      hint: const Text("Choose Branch", style: TextStyle(color: Colors.white38)),
                      isExpanded: true,
                      dropdownColor: const Color(0xFF161625),
                      icon: const Icon(Icons.arrow_drop_down, color: Colors.white54),
                      style: const TextStyle(color: Colors.white),
                      items: ["CSE", "IT", "AI&DS", "CSBS", "ECE", "EEE", "AEI", "MECH", "CIVIL"].map((String branch) {
                        return DropdownMenuItem<String>(value: branch, child: Text(branch));
                      }).toList(),
                      onChanged: (val) => setState(() => _selectedBranch = val),
                    ),
                  ),
                ),
                const SizedBox(height: 25),
              ] else ...[
                _buildTextField("Secret Access Code", _secretCodeController, Icons.vpn_key_outlined, isPassword: true),
                const SizedBox(height: 25),
              ],

              _isLoading
                  ? const CircularProgressIndicator(color: accentColor)
                  : ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: accentColor,
                        minimumSize: const Size(double.infinity, 55),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      ),
                      onPressed: _sendResetOtp,
                      child: const Text("Send Reset Code", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white)),
                    ),
              
              const SizedBox(height: 15),
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text("Back to Login", style: TextStyle(color: Colors.white54)),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _sendResetOtp() async {
      if (_emailController.text.isEmpty || _nameController.text.isEmpty) {
         ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Please fill all required fields")));
         return;
      }
      
      if (_selectedRole == "Student" && _selectedBranch == null) {
         ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Please select your registered branch")));
         return;
      }
      
      if (_selectedRole == "Teacher" && _secretCodeController.text.isEmpty) {
         ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Please enter the Teacher Secret Access Code")));
         return;
      }

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

      setState(() => _isLoading = true);

      try {
        final response = await http.post(
          Uri.parse('${ApiConfig.baseUrl}/auth/forgot-password-otp'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'email': _emailController.text.trim(),
            'username': _nameController.text.trim().toUpperCase(),
            'role': _selectedRole.toLowerCase(),
            'branch': _selectedRole == "Student" ? _selectedBranch : null,
            'secret_code': _selectedRole == "Teacher" ? _secretCodeController.text.trim() : null,
          }),
        );
        
        final body = jsonDecode(response.body);
        
        if (response.statusCode == 200 && mounted) {
          ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Reset Code Sent! Please check your email.")));
          Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => ResetPasswordScreen(email: _emailController.text.trim())));
        } else if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Failed: ${body['detail']}")));
        }
      } catch (e) {
         if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Network Error: $e")));
      } finally {
         if (mounted) setState(() => _isLoading = false);
      }
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
              Text(title, style: TextStyle(color: isSelected ? Colors.white : Colors.white54, fontSize: 13)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTextField(String label, TextEditingController controller, IconData icon, {TextCapitalization textCapitalization = TextCapitalization.none, bool isPassword = false}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(color: Colors.white70, fontSize: 13)),
        const SizedBox(height: 8),
        TextField(
          controller: controller,
          textCapitalization: textCapitalization,
          obscureText: isPassword,
          style: const TextStyle(color: Colors.white),
          decoration: InputDecoration(
            prefixIcon: Icon(icon, color: Colors.white38, size: 20),
            filled: true,
            fillColor: Colors.white.withOpacity(0.05),
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
            hintText: label,
            contentPadding: const EdgeInsets.symmetric(vertical: 15),
            hintStyle: const TextStyle(color: Colors.white24, fontSize: 13),
          ),
        ),
      ],
    );
  }

  @override
  void dispose() {
    _emailController.dispose();
    _nameController.dispose();
    _secretCodeController.dispose();
    super.dispose();
  }
}
