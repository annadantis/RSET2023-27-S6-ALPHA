import 'package:flutter/material.dart';
import '../api_config.dart';

class TeacherMessageNotification extends StatefulWidget {
  final Map<String, dynamic> suggestion;
  final VoidCallback onDismiss;

  const TeacherMessageNotification({
    super.key,
    required this.suggestion,
    required this.onDismiss,
  });

  @override
  State<TeacherMessageNotification> createState() => _TeacherMessageNotificationState();
}

class _TeacherMessageNotificationState extends State<TeacherMessageNotification> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<Offset> _offsetAnimation;
  bool _isMarkingRead = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 500),
      vsync: this,
    );
    _offsetAnimation = Tween<Offset>(
      begin: const Offset(1.5, 0.0),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutBack,
    ));
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _handleMarkRead() async {
    setState(() => _isMarkingRead = true);
    await ApiConfig.markSuggestionRead(widget.suggestion['id']);
    if (mounted) {
      _controller.reverse().then((_) => widget.onDismiss());
    }
  }

  @override
  Widget build(BuildContext context) {
    final String teacherName = widget.suggestion['teacher'] ?? 'A Teacher';
    final String message = widget.suggestion['message'] ?? '';
    final String timestamp = widget.suggestion['timestamp']?.toString().substring(0, 10) ?? '';

    return SlideTransition(
      position: _offsetAnimation,
      child: Material(
        color: Colors.transparent,
        child: Container(
          width: 350,
          margin: const EdgeInsets.all(20),
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: const Color(0xFF1E1E2F),
            borderRadius: BorderRadius.circular(15),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.5),
                blurRadius: 15,
                offset: const Offset(0, 5),
              ),
            ],
            border: Border.all(color: Colors.tealAccent.withOpacity(0.3), width: 1),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                   Row(
                    children: [
                      const SizedBox(width: 8),
                      Text(
                        "Message from $teacherName",
                        style: const TextStyle(color: Colors.tealAccent, fontWeight: FontWeight.bold, fontSize: 12),
                      ),
                    ],
                  ),
                   IconButton(
                    icon: const Text("CLOSE", style: TextStyle(color: Colors.white54, fontSize: 10, fontWeight: FontWeight.bold)),
                    padding: EdgeInsets.zero,
                    constraints: const BoxConstraints(),
                    onPressed: () => _controller.reverse().then((_) => widget.onDismiss()),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text(
                message,
                style: const TextStyle(color: Colors.white, fontSize: 14, height: 1.4),
                maxLines: 4,
                overflow: TextOverflow.ellipsis,
              ),
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    timestamp,
                    style: const TextStyle(color: Colors.white38, fontSize: 10),
                  ),
                   _isMarkingRead 
                    ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.tealAccent))
                    : TextButton(
                        onPressed: _handleMarkRead,
                        style: TextButton.styleFrom(
                          foregroundColor: Colors.tealAccent,
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                        ),
                        child: const Text("Mark as Read", style: TextStyle(fontWeight: FontWeight.bold)),
                      ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
