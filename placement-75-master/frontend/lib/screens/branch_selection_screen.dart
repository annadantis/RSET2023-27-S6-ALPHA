import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';
import 'dashboard_screen.dart';

class BranchSelectionScreen extends StatefulWidget {
  const BranchSelectionScreen({super.key});

  @override
  State<BranchSelectionScreen> createState() => _BranchSelectionScreenState();
}

class _BranchSelectionScreenState extends State<BranchSelectionScreen> {
  String? _selectedBranch;
  bool _isLoading = false;

  final List<Map<String, dynamic>> branches = [
    {'code': 'CSE', 'name': 'Computer Science Engineering', 'icon': Icons.computer},
    {'code': 'IT', 'name': 'Information Technology', 'icon': Icons.devices},
    {'code': 'AIDS', 'name': 'AI & Data Science', 'icon': Icons.psychology},
    {'code': 'CSBS', 'name': 'Computer Science & Business Systems', 'icon': Icons.business},
    {'code': 'ECE', 'name': 'Electronics & Communication', 'icon': Icons.electrical_services},
    {'code': 'EEE', 'name': 'Electrical & Electronics', 'icon': Icons.bolt},
    {'code': 'AEI', 'name': 'Applied Electronics & Instrumentation', 'icon': Icons.electrical_services},
    {'code': 'MECH', 'name': 'Mechanical Engineering', 'icon': Icons.precision_manufacturing},
    {'code': 'CIVIL', 'name': 'Civil Engineering', 'icon': Icons.foundation},
  ];

  Future<void> _submitBranch() async {
    if (_selectedBranch == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Please select your branch")),
      );
      return;
    }

    setState(() => _isLoading = true);

    final success = await Provider.of<AuthProvider>(context, listen: false)
        .updateBranch(_selectedBranch!);

    setState(() => _isLoading = false);

    if (success && context.mounted) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => const DashboardScreen()),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Failed to update branch")),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    const Color scaffoldBg = Color(0xFF0F0C29);
    const Color accentColor = Color(0xFF2196F3);

    return Scaffold(
      backgroundColor: scaffoldBg,
      body: Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 600),
          padding: const EdgeInsets.all(40),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.school, size: 80, color: accentColor),
              const SizedBox(height: 20),
              const Text(
                "Select Your Branch",
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 10),
              const Text(
                "Choose your engineering branch to get personalized technical questions",
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 16, color: Colors.white70),
              ),
              const SizedBox(height: 40),
              
              // Branch Grid
              Expanded(
                child: GridView.builder(
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 2,
                    crossAxisSpacing: 15,
                    mainAxisSpacing: 15,
                    childAspectRatio: 1.5,
                  ),
                  itemCount: branches.length,
                  itemBuilder: (context, index) {
                    final branch = branches[index];
                    final isSelected = _selectedBranch == branch['code'];
                    
                    return GestureDetector(
                      onTap: () => setState(() => _selectedBranch = branch['code']),
                      child: Container(
                        decoration: BoxDecoration(
                          color: isSelected
                              ? accentColor.withOpacity(0.2)
                              : Colors.white.withOpacity(0.05),
                          border: Border.all(
                            color: isSelected ? accentColor : Colors.white12,
                            width: 2,
                          ),
                          borderRadius: BorderRadius.circular(15),
                        ),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              branch['icon'],
                              size: 40,
                              color: isSelected ? accentColor : Colors.white54,
                            ),
                            const SizedBox(height: 10),
                            Text(
                              branch['code'],
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: isSelected ? Colors.white : Colors.white70,
                              ),
                            ),
                            const SizedBox(height: 5),
                            Padding(
                              padding: const EdgeInsets.symmetric(horizontal: 10),
                              child: Text(
                                branch['name'],
                                textAlign: TextAlign.center,
                                style: TextStyle(
                                  fontSize: 11,
                                  color: isSelected ? Colors.white70 : Colors.white38,
                                ),
                                maxLines: 2,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),
              
              const SizedBox(height: 30),
              
              // Continue Button
              SizedBox(
                width: double.infinity,
                height: 55,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: accentColor,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  onPressed: _isLoading ? null : _submitBranch,
                  child: _isLoading
                      ? const CircularProgressIndicator(color: Colors.white)
                      : const Text(
                          "Continue",
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
