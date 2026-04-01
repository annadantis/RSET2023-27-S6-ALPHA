import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:ui';

class PremiumLoadingOverlay extends StatefulWidget {
  final String? message;
  const PremiumLoadingOverlay({super.key, this.message});

  @override
  State<PremiumLoadingOverlay> createState() => _PremiumLoadingOverlayState();
}

class _PremiumLoadingOverlayState extends State<PremiumLoadingOverlay> with TickerProviderStateMixin {
  late AnimationController _fadeController;
  late AnimationController _shimmerController;
  int _tipIndex = 0;
  Timer? _tipTimer;

  final List<String> _tips = [
    "Did you know? Consistent practice improves placement chances by 40%.",
    "Tip: Maintain eye contact with the camera to show confidence.",
    "Pro Tip: Speak at a moderate pace (140 WPM) for best AI evaluation.",
    "Fun Fact: Mock interviews reduce real world anxiety significantly.",
    "Remember: Strong technical fundamentals are the key to any coding round.",
    "Tip: Avoid filler words like 'um' and 'like' for a higher voice score.",
    "Preparing your personalized career roadmap...",
    "Analyzing your performance metrics...",
  ];

  @override
  void initState() {
    super.initState();
    _fadeController = AnimationController(vsync: this, duration: const Duration(seconds: 2))..repeat(reverse: true);
    _shimmerController = AnimationController(vsync: this, duration: const Duration(seconds: 3))..repeat();
    
    _tipTimer = Timer.periodic(const Duration(seconds: 4), (timer) {
      if (mounted) {
        setState(() => _tipIndex = (_tipIndex + 1) % _tips.length);
      }
    });
  }

  @override
  void dispose() {
    _fadeController.dispose();
    _shimmerController.dispose();
    _tipTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0A12), // Deeper, more professional dark
      body: Stack(
        children: [
          // Subtle professional Background mesh
          Positioned.fill(
            child: Opacity(
              opacity: 0.4,
              child: CustomPaint(
                painter: MeshPainter(_fadeController),
              ),
            ),
          ),
          
          Center(
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 15, sigmaY: 15),
              child: Container(
                width: 350,
                padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 50),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.02),
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(color: Colors.white.withOpacity(0.08), width: 1),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const SizedBox(height: 20),
                    
                    // Loading Message with Professional Shimmer
                    AnimatedBuilder(
                      animation: _shimmerController,
                      builder: (context, child) {
                        return ShaderMask(
                          shaderCallback: (bounds) {
                            return LinearGradient(
                              colors: [Colors.white38, Colors.white, Colors.white38],
                              stops: [
                                (_shimmerController.value - 0.2).clamp(0.0, 1.0),
                                _shimmerController.value,
                                (_shimmerController.value + 0.2).clamp(0.0, 1.0),
                              ],
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                            ).createShader(bounds);
                          },
                          child: Text(
                            widget.message ?? "ANALYZING YOUR PROGRESS",
                            textAlign: TextAlign.center,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 15,
                              fontWeight: FontWeight.w800,
                              letterSpacing: 3.0,
                            ),
                          ),
                        );
                      },
                    ),
                    const SizedBox(height: 24),
                    
                    // Ultra-slim minimal progress bar
                    Container(
                      height: 1,
                      width: 140,
                      color: Colors.white12,
                      child: Align(
                        alignment: Alignment.centerLeft,
                        child: AnimatedBuilder(
                          animation: _shimmerController,
                          builder: (context, child) {
                            return Container(
                              width: 140 * _shimmerController.value,
                              height: 1,
                              color: Colors.blueAccent,
                            );
                          },
                        ),
                      ),
                    ),
                    const SizedBox(height: 48),
                    
                    // Minimal Tips Section
                    AnimatedSwitcher(
                      duration: const Duration(milliseconds: 800),
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 10),
                        child: Text(
                          _tips[_tipIndex].toUpperCase(),
                          key: ValueKey(_tipIndex),
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.4),
                            fontSize: 10,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 1.5,
                            height: 1.6,
                          ),
                        ),
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
}

class MeshPainter extends CustomPainter {
  final Animation<double> animation;
  MeshPainter(this.animation) : super(repaint: animation);

  @override
  void paint(Canvas canvas, Size size) {
    final paint1 = Paint()
      ..color = Colors.blueAccent.withOpacity(0.15)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 100);
    
    final paint2 = Paint()
      ..color = Colors.purpleAccent.withOpacity(0.1)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 80);

    canvas.drawCircle(
      Offset(size.width * 0.2 + (animation.value * 50), size.height * 0.3),
      200, paint1);
    
    canvas.drawCircle(
      Offset(size.width * 0.8, size.height * 0.7 - (animation.value * 50)),
      250, paint2);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
