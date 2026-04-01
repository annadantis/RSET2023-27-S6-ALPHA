import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';

class BranchSelectionDialog extends StatefulWidget {
  final String? initialBranch;
  final bool practiceMode; // If true, just return the selection, don't update DB
  const BranchSelectionDialog({super.key, this.initialBranch, this.practiceMode = false});

  @override
  State<BranchSelectionDialog> createState() => _BranchSelectionDialogState();
}

class _BranchSelectionDialogState extends State<BranchSelectionDialog> {
  String? _selectedBranch;
  final List<String> _branches = ["CSE", "IT", "AI&DS", "CSBS", "ECE", "EEE", "AEI", "MECH", "CIVIL"];
  bool _isUpdating = false;

  @override
  void initState() {
    super.initState();
    _selectedBranch = widget.initialBranch;
  }

  @override
  Widget build(BuildContext context) {
    const Color accentColor = Color(0xFF6C63FF);
    const Color dialogBg = Color(0xFF161625);

    return AlertDialog(
      backgroundColor: dialogBg,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      title: Text(
        widget.practiceMode ? "Select Practice Branch" : "Select Your Branch",
        style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
           Text(
            widget.practiceMode 
                ? "Choose a branch to practice questions from. Scores for other branches won't affect your main progress."
                : "Please select your academic branch to personalize your technical questions.",
            style: const TextStyle(color: Colors.white70, fontSize: 13),
          ),
          const SizedBox(height: 20),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.05),
              borderRadius: BorderRadius.circular(12),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<String>(
                value: _selectedBranch,
                hint: const Text("Choose Branch", style: TextStyle(color: Colors.white54)),
                isExpanded: true,
                dropdownColor: dialogBg,
                icon: const Icon(Icons.arrow_drop_down, color: Colors.white54),
                style: const TextStyle(color: Colors.white),
                items: _branches.map((String branch) {
                  return DropdownMenuItem<String>(
                    value: branch,
                    child: Text(branch),
                  );
                }).toList(),
                onChanged: _isUpdating ? null : (String? newValue) {
                  setState(() {
                    _selectedBranch = newValue;
                  });
                },
              ),
            ),
          ),
        ],
      ),
      actions: [
        if (_isUpdating)
          const Center(child: Padding(padding: EdgeInsets.all(8.0), child: CircularProgressIndicator()))
        else
          TextButton(
            onPressed: _selectedBranch == null ? null : _handleUpdate,
            style: TextButton.styleFrom(
              backgroundColor: _selectedBranch == null ? Colors.grey.withOpacity(0.1) : accentColor,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
            ),
            child: const Text("Confirm Selection"),
          ),
      ],
    );
  }

  Future<void> _handleUpdate() async {
    if (_selectedBranch == null) return;

    // If Practice Mode, just return the value
    if (widget.practiceMode) {
      Navigator.pop(context, _selectedBranch);
      return;
    }

    setState(() => _isUpdating = true);
    
    final auth = Provider.of<AuthProvider>(context, listen: false);
    
    print("🔄 Attempting to update branch to: $_selectedBranch");
    
    final success = await auth.updateBranch(_selectedBranch!);
    
    if (mounted) {
      setState(() => _isUpdating = false);
      if (success) {
        Navigator.pop(context, _selectedBranch); // Return the branch string on success too
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Branch set to $_selectedBranch successfully!")),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text("Failed to set branch."),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }
}
