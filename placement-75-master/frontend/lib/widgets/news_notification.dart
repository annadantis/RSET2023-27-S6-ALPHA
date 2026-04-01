import 'package:flutter/material.dart';
import '../api_config.dart';

class NewsNotification extends StatefulWidget {
  final Map<String, dynamic> item;
  final VoidCallback onDismiss;
  final Function(String) onRead;

  const NewsNotification({
    super.key,
    required this.item,
    required this.onDismiss,
    required this.onRead,
  });

  @override
  State<NewsNotification> createState() => _NewsNotificationState();
}

class _NewsNotificationState extends State<NewsNotification> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<Offset> _offsetAnimation;
  String? summary;
  bool loadingSummary = false;

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

  Future<void> _handleRead() async {
    if (summary == null) {
      setState(() => loadingSummary = true);
      try {
        final result = await ApiConfig.fetchNewsSummary(
          widget.item['title'] ?? '',
          url: widget.item['url'],
        );
        setState(() {
          summary = result;
          loadingSummary = false;
        });
      } catch (e) {
        setState(() => loadingSummary = false);
        summary = widget.item['title']; // Fallback
      }
    }
    
    if (summary != null) {
      widget.onRead(summary!);
    }
  }

  @override
  Widget build(BuildContext context) {
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
            border: Border.all(color: Colors.purpleAccent.withOpacity(0.3), width: 1),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Row(
                    children: [
                      Icon(Icons.trending_up, color: Colors.purpleAccent, size: 18),
                      SizedBox(width: 8),
                      Text(
                        "Industry Update",
                        style: TextStyle(color: Colors.purpleAccent, fontWeight: FontWeight.bold, fontSize: 12),
                      ),
                    ],
                  ),
                  IconButton(
                    icon: const Icon(Icons.close, color: Colors.white54, size: 18),
                    padding: EdgeInsets.zero,
                    constraints: const BoxConstraints(),
                    onPressed: widget.onDismiss,
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text(
                widget.item['title'] ?? 'No Title',
                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 14),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
              if (summary != null) ...[
                const SizedBox(height: 8),
                Text(
                  summary!,
                  style: const TextStyle(color: Colors.white70, fontSize: 12, fontStyle: FontStyle.italic),
                ),
              ],
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      icon: loadingSummary 
                        ? const SizedBox(width: 12, height: 12, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.purpleAccent))
                        : const Icon(Icons.volume_up, size: 16),
                      label: Text(loadingSummary ? "Summarizing..." : "Read Summary"),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: Colors.purpleAccent,
                        side: const BorderSide(color: Colors.purpleAccent),
                        padding: const EdgeInsets.symmetric(vertical: 8),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                      ),
                      onPressed: loadingSummary ? null : _handleRead,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.white.withOpacity(0.1),
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 8),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                      ),
                      onPressed: () {
                        ApiConfig.stopSpeech();
                        widget.onDismiss();
                      },
                      child: const Text("Got it"),
                    ),
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
