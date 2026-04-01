import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:record/record.dart';
import 'package:http/http.dart' as http;
import 'package:provider/provider.dart';
import 'package:path_provider/path_provider.dart';
import 'package:placement_assistant/widgets/loading_overlay.dart';
import '../api_config.dart';
import '../providers/auth_provider.dart';

class InterviewScreen extends StatefulWidget {
  const InterviewScreen({super.key});
  @override
  State<InterviewScreen> createState() => _InterviewScreenState();
}

class _InterviewScreenState extends State<InterviewScreen> with WidgetsBindingObserver {
  CameraController? _camera;
  final AudioRecorder _recorder = AudioRecorder();
  List<dynamic> _questions = [];
  int _currentIndex = 0;
  int _secondsLeft = 30;
  Timer? _timer;
  Timer? _frameTimer;
  bool _isLoading = false;
  Map<String, dynamic>? _result;
  String? _liveWarning = "INITIALIZING";
  String? _cameraError;
  bool _screenWarning = false;
  bool _isTestStarted = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _startSession();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused || state == AppLifecycleState.inactive || state == AppLifecycleState.hidden) {
      if (!_isLoading && _result == null) {
        setState(() => _screenWarning = true);
        Future.delayed(const Duration(seconds: 4), () {
          if (mounted) setState(() => _screenWarning = false);
        });
      }
    }
  }

  Future<void> _startSession() async {
    try {
      final cams = await availableCameras();
      if (cams.isEmpty) {
        setState(() => _cameraError = "No camera found. Please connect a webcam.");
        return;
      }
      
      _camera = CameraController(
        cams[0], 
        ResolutionPreset.medium,
        enableAudio: false,
      );
      
      await _camera!.initialize();
      if (mounted) setState(() => _cameraError = null);
    } catch (e) {
      if (mounted) {
        setState(() {
          _cameraError = "Camera Error: Please ensure no other app is using the webcam.";
          print("Camera Init Error: $e");
        });
      }
    }
    
    try {
      final auth = Provider.of<AuthProvider>(context, listen: false);
      final response = await http.get(Uri.parse('${auth.baseUrl}/get_questions/${auth.username}/INTERVIEW'));
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (mounted) {
          setState(() {
            _questions = data['questions'];
            _currentIndex = 0; 
          });
        }
        if (_cameraError == null) _startLiveFrameAnalysis();
      }
    } catch (e) {
      print("Error fetching questions: $e");
    }
  }

  void _beginInterview() async {
    setState(() {
      _isTestStarted = true;
    });
    
    // Start session-wide video recording for behavioral analysis
    if (_camera != null && _camera!.value.isInitialized) {
      try {
        await _camera!.startVideoRecording();
        print("DEBUG: Session video recording started.");
      } catch (e) {
        print("DEBUG: Error starting video recording: $e");
      }
    }
    
    _startQuestionTimer();
  }

  void _startLiveFrameAnalysis() {
    _frameTimer = Timer.periodic(const Duration(milliseconds: 1500), (timer) async {
      if (_camera == null || !_camera!.value.isInitialized || _isLoading) return;
      if (_camera!.value.isTakingPicture) return;
      
      try {
        final xFile = await _camera!.takePicture();
        final auth = Provider.of<AuthProvider>(context, listen: false);
        var req = http.MultipartRequest('POST', Uri.parse('${auth.baseUrl}/process_frame'));
        req.fields['username'] = auth.username!;
        req.files.add(await http.MultipartFile.fromPath('frame', xFile.path));
        
        final res = await req.send();
        if (res.statusCode == 200) {
          final body = await http.Response.fromStream(res);
          final data = jsonDecode(body.body);
          if (mounted) {
            setState(() {
              _liveWarning = data['warning'];
            });
            print("DEBUG: Live Warning from Backend: $_liveWarning");
          }
        }
        // Cleanup temp file to avoid filling up disk
        File(xFile.path).delete().catchError((e) => null);
      } catch (e) {
        // Ignore frame dropping errors
      }
    });
  }

  void _startQuestionTimer() async {
    setState(() => _secondsLeft = 30);
    final path = (await getTemporaryDirectory()).path;
    await _recorder.start(const RecordConfig(), path: '$path/q_$_currentIndex.m4a');

    _timer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (_secondsLeft > 0) {
        setState(() => _secondsLeft--);
      } else {
        _timer?.cancel();
        _nextOrSubmit();
      }
    });
  }

  void _nextOrSubmit() async {
    final path = await _recorder.stop();
    final auth = Provider.of<AuthProvider>(context, listen: false);
    
    if (_questions.isEmpty || _currentIndex >= _questions.length) return;

    var stepReq = http.MultipartRequest('POST', Uri.parse('${auth.baseUrl}/evaluate_step'));
    stepReq.fields['username'] = auth.username!;
    stepReq.fields['question'] = _questions[_currentIndex]['question'];
    stepReq.fields['index'] = _currentIndex.toString();
    stepReq.files.add(await http.MultipartFile.fromPath('audio', path!));
    await stepReq.send();

    if (_currentIndex < _questions.length - 1) {
      setState(() {
        _currentIndex++;
      });
      _startQuestionTimer();
    } else {
      _uploadFinalData();
    }
  }

  Future<void> _uploadFinalData() async {
    setState(() {
      _isLoading = true;
      _liveWarning = null;
    });
    _frameTimer?.cancel();
    
    XFile? videoFile;
    if (_camera != null && _camera!.value.isRecordingVideo) {
      try {
        videoFile = await _camera!.stopVideoRecording();
        print("DEBUG: Session video recording stopped: ${videoFile.path}");
      } catch (e) {
        print("DEBUG: Error stopping video recording: $e");
      }
    }

    final auth = Provider.of<AuthProvider>(context, listen: false);
    // Request final report with the recorded video!
    var req = http.MultipartRequest('POST', Uri.parse('${auth.baseUrl}/final_session_report'));
    req.fields['username'] = auth.username!;
    
    if (videoFile != null) {
      req.files.add(await http.MultipartFile.fromPath('video', videoFile.path));
      print("DEBUG: Attaching video file to final report.");
    } else {
      print("DEBUG: Video file is null, skipping attachment.");
    }
    
    var res = await req.send();
    var body = await http.Response.fromStream(res);
    print("DEBUG: Final Report Status: ${res.statusCode}");
    
    if (mounted) {
      setState(() {
        _result = jsonDecode(body.body);
        _isLoading = false;
      });
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _timer?.cancel();
    _frameTimer?.cancel();
    _camera?.dispose();
    _recorder.dispose();
    super.dispose();
  }

  String _getWarningMessage() {
    if (_screenWarning) return "⚠️  WARNING: Screen change detected! Stay on this screen.";
    switch (_liveWarning) {
      case "INITIALIZING": return "🔄 Initializing Camera... Please wait.";
      case "FACE_NOT_DETECTED": return "🤷 Face not detected! Move into view.";
      case "OUT_OF_BOX": return "📦 Face out of frame! Center yourself.";
      case "EYE_GAZE": return "👀 Looking away! Maintain eye contact.";
      case "LOW_LIGHT": return "💡 Too dark! Please improve your lighting.";
      case "MULTIPLE_FACES": return "👥 Multiple people detected! Ensure you are alone.";
      case "LOOKING_GREAT": return "✅ Perfect! Keep it up.";
      case "MEDIAPIPE_ERROR": return "⚠️ AI Error: MediaPipe is unavailable on server.";
      default: return "";
    }
  }

  Color _getWarningColor() {
    if (_screenWarning) return Colors.redAccent.shade700;
    switch (_liveWarning) {
      case "INITIALIZING": return Colors.blueAccent;
      case "FACE_NOT_DETECTED": return Colors.red;
      case "OUT_OF_BOX": return Colors.orange;
      case "EYE_GAZE": return Colors.amber.shade700;
      case "LOW_LIGHT": return Colors.deepPurpleAccent;
      case "MULTIPLE_FACES": return Colors.redAccent;
      case "LOOKING_GREAT": return Colors.greenAccent.shade700;
      case "MEDIAPIPE_ERROR": return Colors.redAccent.shade400;
      default: return Colors.transparent;
    }
  }

  Future<bool> _onWillPop() async {
    if (_result != null || (!_isTestStarted && !_isLoading)) return true;

    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF161625),
        title: const Text("Exit Interview?", style: TextStyle(color: Colors.white)),
        content: const Text(
          "⚠️ Your current interview progress will NOT be saved. The result will not be recorded in your analytics.\n\nAre you sure you want to leave?",
          style: TextStyle(color: Colors.white70, height: 1.5),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text("Continue", style: TextStyle(color: Colors.cyanAccent)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text("Leave", style: TextStyle(color: Colors.redAccent)),
          ),
        ],
      ),
    );
    return confirm ?? false;
  }

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: _onWillPop,
      child: Scaffold(
        extendBodyBehindAppBar: true,
        appBar: AppBar(
          backgroundColor: Colors.transparent,
          elevation: 0,
          leading: IconButton(
            icon: const Icon(Icons.arrow_back_ios_new, color: Colors.white, size: 20),
            onPressed: () async {
              if (await _onWillPop()) Navigator.pop(context);
            },
          ),
          title: const Text("AI Interview", style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
          centerTitle: true,
        ),
        body: Container(
          color: const Color(0xFF0F0C29), // Match main app Scaffold background
          child: SafeArea(
            child: _isLoading 
              ? _buildLoadingView()
              : _result != null ? _buildReportView() : _buildInterviewView(),
          ),
        ),
      ),
    );
  }

  Widget _buildLoadingView() {
    return PremiumLoadingOverlay(message: "Analyzing and Generating the  Report...");
  }

  Widget _buildInterviewView() {
    if (_questions.isEmpty) {
      return PremiumLoadingOverlay(message: "Preparing Your Interview Questions...");
    }

    final warningMsg = _getWarningMessage();
    // Only block the start button for ACTUAL warnings (not LOOKING_GREAT or empty)
    final isActualWarning = _liveWarning != null &&
        _liveWarning != "LOOKING_GREAT" &&
        _liveWarning != "INITIALIZING" &&
        _liveWarning!.isNotEmpty &&
        !_screenWarning;
    final hasWarning = _screenWarning || isActualWarning;

    if (_cameraError != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(30.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.videocam_off, color: Colors.redAccent, size: 64),
              const SizedBox(height: 20),
              Text(
                _cameraError!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white, fontSize: 16),
              ),
              const SizedBox(height: 30),
              ElevatedButton(
                onPressed: _startSession,
                style: ElevatedButton.styleFrom(backgroundColor: Colors.cyanAccent),
                child: const Text("Retry Connection", style: TextStyle(color: Colors.black)),
              )
            ],
          ),
        ),
      );
    }

    return Column(
      children: [
        // Camera Preview with Modern Styling
        if (_camera != null && _camera!.value.isInitialized)
          Expanded(
            flex: 5,
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(24),
                boxShadow: [
                  BoxShadow(color: hasWarning ? _getWarningColor().withOpacity(0.5) : Colors.cyanAccent.withOpacity(0.1), blurRadius: 20, spreadRadius: 2)
                ],
                border: Border.all(
                  color: hasWarning ? _getWarningColor() : Colors.white12, 
                  width: hasWarning ? 3 : 1
                )
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(24),
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    CameraPreview(_camera!),
                    // Live Status Overlay — shown for all states, red only for actual warnings
                    if (warningMsg.isNotEmpty)
                      Positioned(
                        top: 20,
                        left: 20,
                        right: 20,
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: BackdropFilter(
                            filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                            child: Container(
                              padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
                              color: hasWarning
                                  ? _getWarningColor().withOpacity(0.8)
                                  : Colors.greenAccent.withOpacity(0.6),
                              child: Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Expanded(
                                    child: Text(
                                      warningMsg,
                                      textAlign: TextAlign.center,
                                      style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 15),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ),
          
        // Question & Control Card OR Pre-Interview Card
        Expanded(
          flex: 4,
          child: Container(
            margin: const EdgeInsets.fromLTRB(20, 10, 20, 20),
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: const Color(0xFF161625), // Matched card color
              borderRadius: BorderRadius.circular(30),
            ),
            child: _isTestStarted 
              ? Column(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text("Question ${_currentIndex + 1}/${_questions.length}", 
                          style: const TextStyle(color: Colors.cyanAccent, fontWeight: FontWeight.w600, fontSize: 14)),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                          decoration: BoxDecoration(
                            color: _secondsLeft < 10 ? Colors.redAccent.withOpacity(0.2) : Colors.greenAccent.withOpacity(0.2),
                            borderRadius: BorderRadius.circular(20)
                          ),
                          child: Row(
                            children: [
                              Icon(Icons.timer, size: 16, color: _secondsLeft < 10 ? Colors.redAccent : Colors.greenAccent),
                              const SizedBox(width: 6),
                              Text(
                                "00:${_secondsLeft.toString().padLeft(2, '0')}", 
                                style: TextStyle(
                                  color: _secondsLeft < 10 ? Colors.redAccent : Colors.greenAccent, 
                                  fontWeight: FontWeight.bold
                                )
                              ),
                            ],
                          ),
                        )
                      ],
                    ),
                    
                    Text(
                      _questions[_currentIndex]['question'],
                      textAlign: TextAlign.center, 
                      style: const TextStyle(color: Colors.white, fontSize: 18, height: 1.4, fontWeight: FontWeight.w500),
                    ),
                    
                    // Audio Wave Animation Placeholder
                    Container(
                      height: 60,
                      width: double.infinity,
                      decoration: BoxDecoration(
                        color: Colors.black12,
                        borderRadius: BorderRadius.circular(30)
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.mic, 
                            color: _secondsLeft % 2 == 0 ? Colors.cyanAccent : Colors.white54, 
                            size: 28
                          ),
                          const SizedBox(width: 15),
                          const Text("Recording Answer...", style: TextStyle(color: Colors.white54, fontSize: 14, letterSpacing: 1.2)),
                        ],
                      ),
                    ),
                    
                    const SizedBox(height: 10),
                    
                    OutlinedButton.icon(
                      onPressed: () {
                        _timer?.cancel();
                        _nextOrSubmit();
                      },
                      icon: Icon(
                        _currentIndex < _questions.length - 1 ? Icons.skip_next : Icons.check_circle, 
                        color: Colors.cyanAccent, 
                        size: 20
                      ),
                      label: Text(
                        _currentIndex < _questions.length - 1 ? "Next Question" : "Finish Interview", 
                        style: const TextStyle(color: Colors.cyanAccent, fontWeight: FontWeight.bold)
                      ),
                      style: OutlinedButton.styleFrom(
                        side: BorderSide(color: Colors.cyanAccent.withOpacity(0.5)),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                      ),
                    ),
                  ],
                )
              : Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text(
                      "Pre-Interview Check",
                      style: TextStyle(color: Colors.cyanAccent, fontSize: 24, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 20),
                    const Text(
                      "Please ensure your face is clearly visible, centered in the camera, and maintain eye contact to begin.",
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.white70, fontSize: 16, height: 1.5),
                    ),
                    const SizedBox(height: 30),
                    SizedBox(
                      width: double.infinity,
                      height: 55,
                      child: ElevatedButton(
                        onPressed: hasWarning ? null : _beginInterview,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.cyanAccent.withOpacity(0.8),
                          disabledBackgroundColor: Colors.grey.withOpacity(0.3),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                        ),
                        child: Text(
                          hasWarning ? "Fix Warnings to Start" : "Start Interview",
                          style: TextStyle(
                            color: hasWarning ? Colors.white54 : Colors.black,
                            fontSize: 18,
                            fontWeight: FontWeight.bold
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
          ),
        ),
      ],
    );
  }

  Widget _buildReportView() {
    final scores = _result!['individual_scores'] ?? {};
    final fillerCount = _result!['filler_words_count'] ?? 0;
    
    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 30),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 1. Final Grade Glass Card
          Center(
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 30, horizontal: 40),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    Colors.cyanAccent.withOpacity(0.15),
                    Colors.purpleAccent.withOpacity(0.15),
                  ],
                ),
                borderRadius: BorderRadius.circular(32),
                border: Border.all(color: Colors.white.withOpacity(0.1), width: 1.5),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.3),
                    blurRadius: 40,
                    offset: const Offset(0, 20),
                  )
                ],
              ),
              child: Column(
                children: [
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.05),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(Icons.stars, color: Colors.amberAccent, size: 32),
                  ),
                  const SizedBox(height: 15),
                  const Text("Final Session Performance", style: TextStyle(color: Colors.white70, fontSize: 16, letterSpacing: 1.2, fontWeight: FontWeight.w500)),
                  const SizedBox(height: 8),
                  Text("${_result!['final_score']}", style: const TextStyle(color: Colors.white, fontSize: 64, fontWeight: FontWeight.w900, letterSpacing: -1)),
                ],
              ),
            ),
          ),
          const SizedBox(height: 40),

          // 2. Metrics Grid
          Row(
            children: [
              _buildMetricCard("TECHNICAL", "${scores['technical'] ?? 'N/A'}", Icons.code, Colors.cyanAccent),
              const SizedBox(width: 12),
              _buildMetricCard("COMMUNICATION", "${scores['voice'] ?? 'N/A'}", Icons.mic, Colors.purpleAccent),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              _buildMetricCard("CAMERA", "${scores['camera'] ?? 'N/A'}", Icons.videocam, Colors.blueAccent),
              const SizedBox(width: 12),
              _buildMetricCard("FILLER WORDS", "$fillerCount", Icons.format_quote, Colors.orangeAccent),
            ],
          ),
          const SizedBox(height: 12),
          
          // New: Session-wide behavioral metrics
          if (_result!['session_metrics'] != null)
            Row(
              children: [
                _buildMetricCard("EYE CONTACT", "${_result!['session_metrics']['eye_contact'] ?? '0'}%", Icons.visibility, Colors.tealAccent),
                const SizedBox(width: 12),
                _buildMetricCard("AVG SPEECH RATE", "${_result!['session_metrics']['avg_wpm'] ?? 0} WPM", Icons.speed, Colors.greenAccent),
              ],
            ),
          const SizedBox(height: 40),

          // 3. Narrative Feedback
          _buildPremiumFeedbackSection("Critical Confidence Analysis", _result!['overall_confidence'], Icons.insights, [Colors.cyanAccent, Colors.blueAccent]),
          const SizedBox(height: 24),
          _buildPremiumFeedbackSection("Behavioral & Speech Observations", _result!['behavioral_feedback'], Icons.psychology, [Colors.purpleAccent, Colors.deepPurpleAccent]),
          
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 40),
            child: Row(
              children: [
                Expanded(child: Divider(color: Colors.white10, thickness: 1.5)),
                Padding(
                  padding: EdgeInsets.symmetric(horizontal: 16),
                  child: Text("QUESTION BREAKDOWN", style: TextStyle(color: Colors.white38, fontSize: 14, fontWeight: FontWeight.w800, letterSpacing: 2)),
                ),
                Expanded(child: Divider(color: Colors.white10, thickness: 1.5)),
              ],
            ),
          ),

          // 4. Detailed Questions
          ...(_result!['technical_report'] as List).map((item) {
            final vStats = item['voice_stats'];
            return Container(
              margin: const EdgeInsets.only(bottom: 16),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.02),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Colors.white.withOpacity(0.05)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                        decoration: BoxDecoration(color: Colors.cyanAccent.withOpacity(0.1), borderRadius: BorderRadius.circular(6)),
                        child: Text(item['accuracy'] ?? "0%", style: const TextStyle(color: Colors.cyanAccent, fontSize: 12, fontWeight: FontWeight.bold)),
                      ),
                      const SizedBox(width: 10),
                      Expanded(child: Text(item['question'] ?? "", style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600, fontSize: 16, height: 1.3))),
                    ],
                  ),
                  const SizedBox(height: 16),
                  
                  // Voice Stats for this question
                  if (vStats != null)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 16),
                      child: Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.purpleAccent.withOpacity(0.05),
                          borderRadius: BorderRadius.circular(15),
                          border: Border.all(color: Colors.purpleAccent.withOpacity(0.1)),
                        ),
                        child: Row(
                          children: [
                            const Icon(Icons.record_voice_over, color: Colors.purpleAccent, size: 16),
                            const SizedBox(width: 12),
                            Text("${vStats['wpm']} WPM", style: const TextStyle(color: Colors.white70, fontSize: 13, fontWeight: FontWeight.bold)),
                            const SizedBox(width: 15),
                            const Icon(Icons.format_quote, color: Colors.orangeAccent, size: 14),
                            const SizedBox(width: 6),
                            Text("${vStats['fillers']} fillers", style: const TextStyle(color: Colors.white70, fontSize: 13)),
                            const Spacer(),
                            Text(vStats['feedback'], style: const TextStyle(color: Colors.white38, fontSize: 11, fontStyle: FontStyle.italic)),
                          ],
                        ),
                      ),
                    ),
                    
                  _buildModernResponseBox("YOUR RESPONSE", item['user_input'] ?? item['your_answer'] ?? "N/A", Colors.white70),
                  const SizedBox(height: 12),
                  _buildModernResponseBox("IDEAL ANSWER", item['ideal_answer'] ?? "N/A", Colors.greenAccent.shade400),
                  const SizedBox(height: 12),
                  _buildModernResponseBox("IMPROVEMENTS", item['improvement'] ?? "N/A", Colors.orangeAccent),
                ],
              ),
            );
          }).toList(),
          const SizedBox(height: 40),
        ],
      ),
    );
  }

  Widget _buildMetricCard(String label, String value, IconData icon, Color color) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
        decoration: BoxDecoration(
          color: color.withOpacity(0.05),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: color.withOpacity(0.12)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: color, size: 20),
            const SizedBox(height: 10),
            Text(value, style: TextStyle(color: color, fontSize: 24, fontWeight: FontWeight.w800)),
            const SizedBox(height: 4),
            Text(label, style: const TextStyle(color: Colors.white38, fontSize: 14, fontWeight: FontWeight.w900, letterSpacing: 1.0)),
          ],
        ),
      ),
    );
  }

  Widget _buildPremiumFeedbackSection(String title, String content, IconData icon, List<Color> colors) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.02),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.05)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              ShaderMask(
                shaderCallback: (bounds) => LinearGradient(colors: colors).createShader(bounds),
                child: Icon(icon, color: Colors.white, size: 20),
              ),
              const SizedBox(width: 10),
              Text(title, style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold, letterSpacing: 0.4)),
            ],
          ),
          const SizedBox(height: 12),
          Text(content, style: const TextStyle(color: Colors.white70, fontSize: 14, height: 1.5)),
        ],
      ),
    );
  }

  Widget _buildModernResponseBox(String label, String text, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: TextStyle(color: color.withOpacity(0.4), fontSize: 12, fontWeight: FontWeight.w900, letterSpacing: 1.2)),
        const SizedBox(height: 6),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: color.withOpacity(0.04),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Text(text, style: TextStyle(color: color, fontSize: 16, height: 1.4, fontWeight: FontWeight.w500)),
        ),
      ],
    );
  }

  Widget _buildResponseRow(String label, String text, Color color) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(width: 120, child: Text(label, style: TextStyle(color: color.withOpacity(0.7), fontSize: 13, fontWeight: FontWeight.w500))),
        Expanded(child: Text(text, style: TextStyle(color: color, fontSize: 14, height: 1.3))),
      ],
    );
  }
  
  Widget _buildFeedbackSection(String title, String content, IconData icon) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(icon, color: Colors.cyanAccent, size: 20),
            const SizedBox(width: 10),
            Text(title, style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600)),
          ],
        ),
        const SizedBox(height: 10),
        Text(content, style: const TextStyle(color: Colors.white70, fontSize: 15, height: 1.4)),
      ],
    );
  }

  Widget _buildScoreCard(String label, String score, Color color) {
    return Container(
      width: 140,
      padding: const EdgeInsets.symmetric(vertical: 20),
      decoration: BoxDecoration(
        color: color.withOpacity(0.05),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: color.withOpacity(0.3))
      ),
      child: Column(
        children: [
          Text(score, style: TextStyle(color: color, fontSize: 32, fontWeight: FontWeight.bold)),
          const SizedBox(height: 4),
          Text(label, style: const TextStyle(color: Colors.white54, fontSize: 11, fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }
}