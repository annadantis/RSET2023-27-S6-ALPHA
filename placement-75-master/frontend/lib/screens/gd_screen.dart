import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:provider/provider.dart';
import 'package:web_socket_channel/io.dart';
import 'package:google_fonts/google_fonts.dart';
import '../api_config.dart';
import '../providers/auth_provider.dart';
import '../services/gd_tts_controller.dart';
import '../widgets/loading_overlay.dart';
import 'result_page.dart';

class GdScreen extends StatefulWidget {
  const GdScreen({super.key});

  @override
  State<GdScreen> createState() => _GdScreenState();
}

enum GDState { instructions, discussion, evaluating }

class _GdScreenState extends State<GdScreen> with WidgetsBindingObserver {
  GDState _currentState = GDState.instructions;
  CameraController? _cameraController;
  final AudioRecorder _audioRecorder = AudioRecorder();
  final GDTTSController _tts = GDTTSController();

  IOWebSocketChannel? _wsChannel;
  bool _wsConnected = false; // ← guard against writing to closed sink
  String _activeSpeaker = "Bot_A";
  String _waitingSpeaker = "";
  bool _handRaised = false;
  List<String> _audioPaths = [];

  bool _cameraInitialized = false;
  bool _recording = false;   // true only when USER is speaking live
  bool _sessionRecording = false; // true for the whole session audio capture
  bool _isDisposed = false;
  int _countdown = 45; // Per-turn speaking timer
  Timer? _timer;
  int _sessionCountdown = 180; // Total GD session time (in seconds)
  Timer? _sessionTimer;
  Timer? _frameTimer;        // Live frame analysis timer

  Map<String, dynamic>? topic;
  String _sessionBotContext = "";
  String? _liveWarning;      // Warning code from /process_frame
  bool _screenWarning = false; // True if user switches away
  String? _currentVideoPath;

  // ── WARNING HELPERS ───────────────────────────────────────────────────────

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

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeSystem();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused ||
        state == AppLifecycleState.inactive ||
        state == AppLifecycleState.hidden) {
      if (_currentState == GDState.discussion) {
        setState(() => _screenWarning = true);
        Future.delayed(const Duration(seconds: 4), () {
          if (mounted) setState(() => _screenWarning = false);
        });
      }
    }
  }

  Future<void> _initializeSystem() async {
    _initTTSConfig();
    await fetchTopic();
    await _initCamera();
  }

  void _initTTSConfig() {
    _tts.onQueueEmpty = () {
      if (!_isDisposed && mounted) {
        print("TTS: Queue is empty. Sending TTS_COMPLETED to backend.");
        _wsSend({"type": "TTS_COMPLETED"});
      }
    };
  }

  // ── 1. SESSION START ──────────────────────────────────────────────────────

  void _startDiscussion() {
    setState(() {
      _currentState = GDState.discussion;
      _sessionCountdown = 180;
    });

    _startSessionAudioRecording();
    _startLiveFrameAnalysis();  // Begin live camera warnings

    _sessionTimer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (!mounted || _isDisposed) return;
      if (_sessionCountdown > 0) {
        setState(() => _sessionCountdown--);
      } else {
        t.cancel();
        _finishAndEvaluate();
      }
    });

    _connectWebSocket();
    _startVideoRecording();
  }

  void _startLiveFrameAnalysis() {
    _frameTimer?.cancel();
    _frameTimer = Timer.periodic(const Duration(milliseconds: 1500), (timer) async {
      if (_cameraController == null || !_cameraController!.value.isInitialized) return;
      if (_cameraController!.value.isTakingPicture) return;
      if (_isDisposed || _currentState == GDState.evaluating) return;

      try {
        final xFile = await _cameraController!.takePicture();
        final auth = Provider.of<AuthProvider>(context, listen: false);
        
        // Diagnostic Logging
        print("📸 GD: Sending frame to backend for analysis...");
        
        final req = http.MultipartRequest('POST', Uri.parse('${auth.baseUrl}/process_frame'));
        req.fields['username'] = auth.username ?? 'gd_user';
        req.files.add(await http.MultipartFile.fromPath('frame', xFile.path));

        final res = await req.send();
        if (res.statusCode == 200) {
          final body = await http.Response.fromStream(res);
          final data = jsonDecode(body.body);
          final newWarning = data['warning'] as String?;
          
          if (mounted && !_isDisposed) {
            setState(() => _liveWarning = newWarning);
            if (newWarning != null && newWarning != "LOOKING_GREAT") {
              print("🚨 GD Warning Received: $newWarning");
            }
          }
        } else {
          print("❌ GD Frame Analysis Error: HTTP ${res.statusCode}");
        }
        File(xFile.path).delete().catchError((_) => File(xFile.path));
      } catch (e) {
        print("⚠️ GD Frame Analysis Exception: $e");
      }
    });
  }

  /// Starts a continuous audio recording for the whole GD session.
  Future<void> _startSessionAudioRecording() async {
    // Deprecated: We now record distinct audio chunks per user turn
  }

  // ── 2. WEBSOCKET ──────────────────────────────────────────────────────────

  void _connectWebSocket() {
    final auth = Provider.of<AuthProvider>(context, listen: false);
    final wsBase =
        auth.baseUrl.replaceAll('http://', '').replaceAll('https://', '');

    // Ensure topic_id and username are always in the URL
    final topicId = topic!['topic_id'].toString();
    final username = auth.username;
    final wsUrl = "ws://$wsBase/gd_module/ws/gd_meeting/$topicId?username=$username";

    _wsChannel = IOWebSocketChannel.connect(Uri.parse(wsUrl));
    _wsConnected = true;
    print("WebSocket: Connecting to $wsUrl");

    _wsChannel!.stream.listen(
      (message) {
        if (_isDisposed || !mounted) return;
        final data = jsonDecode(message);
        final type = data['type'] as String;
        print("WebSocket: Received message: $type");
        if (type == 'BOT_RAISE_HAND') {
          final speaker = data['speaker'] as String;
          print("WebSocket: Bot Raised Hand: $speaker");
          setState(() {
            _waitingSpeaker = speaker;
          });
        } else if (type == 'BOT_SAYS') {
          final speaker = data['speaker'] as String;
          final text = data['text'] as String;
          print("WebSocket: Bot Says: $text");

          setState(() {
            _waitingSpeaker = ""; // They are no longer waiting, they are speaking
            _activeSpeaker = speaker;
            
            String friendlyName = speaker == "Bot_Mod" ? "Thomas" : (speaker == "Bot_A" ? "Aravind" : "George");
            _sessionBotContext += "$friendlyName: $text\n";
          });

          // Wrap TTS call in addPostFrameCallback to keep it on the platform thread
          SchedulerBinding.instance.addPostFrameCallback((_) {
            if (!_isDisposed) _tts.speak(text, speaker);
          });
        } else if (type == 'STATUS' && data['command'] == 'GRANT_FLOOR') {
          print("WebSocket: Floor granted to User");
          SchedulerBinding.instance.addPostFrameCallback((_) {
            if (!_isDisposed) {
              _tts.stop();
              _startUserTurn();
            }
          });
        } else if (type == 'SESSION_END') {
          print("WebSocket: Session ended by server");
          // Only trigger if not already evaluating
          if (_currentState != GDState.evaluating) {
            _finishAndEvaluate();
          }
        }
      },
      onError: (err) {
        print("WebSocket Error: $err");
        _wsConnected = false;
      },
      onDone: () {
        print("WebSocket: Connection Closed");
        _wsConnected = false;
      },
    );
  }

  /// Safe wrapper — never writes to a closed sink
  void _wsSend(Map<String, dynamic> payload) {
    if (_wsConnected && _wsChannel != null) {
      try {
        _wsChannel!.sink.add(jsonEncode(payload));
      } catch (e) {
        print("WebSocket send error: $e");
        _wsConnected = false;
      }
    }
  }

  // ── 3. USER TURN LOGIC ───────────────────────────────────────────────────

  void _toggleHand() async {
    if (!_wsConnected) return; 
    
    if (_handRaised) {
      // Lowering hand
      setState(() => _handRaised = false);
      final path = await _stopUserTurnTimer();
      if (path != null) {
        // Process turn (transcription) so the next bot can respond to it
        await _uploadUserTurn(path);
      }
      _wsSend({"type": "LOWER_HAND"});
    } else {
      // Raising hand
      setState(() => _handRaised = true);
      _wsSend({"type": "RAISE_HAND"});
    }
  }

  /// Called when the backend grants the floor (after RAISE_HAND)
  Future<void> _startUserTurn() async {
    if (_recording) return;
    
    try {
      final dir = await getTemporaryDirectory();
      final path = "${dir.path}/gd_turn_${DateTime.now().millisecondsSinceEpoch}.wav";
      await _audioRecorder.start(
        const RecordConfig(encoder: AudioEncoder.wav),
        path: path,
      );
    } catch (e) {
      print("Audio start error: $e");
    }
    setState(() {
      _recording = true;
      _activeSpeaker = "User";
    });
  }

  Future<String?> _stopUserTurnTimer() async {
    String? path;
    if (await _audioRecorder.isRecording()) {
      path = await _audioRecorder.stop();
      if (path != null) _audioPaths.add(path);
    }
    setState(() {
      _recording = false;
    });
    return path;
  }

  Future<void> _uploadUserTurn(String path) async {
    final auth = Provider.of<AuthProvider>(context, listen: false);
    final url = Uri.parse("${auth.baseUrl}/gd_module/add_user_turn");

    var request = http.MultipartRequest('POST', url);
    request.fields['username'] = auth.username ?? 'anonymous';
    request.files.add(await http.MultipartFile.fromPath('audio', path));

    print("GD: Uploading audio turn for real-time transcription...");
    try {
      final response = await request.send();
      if (response.statusCode == 200) {
        final respBody = await response.stream.bytesToString();
        final data = jsonDecode(respBody);
        final transcript = data['transcript'] ?? "";
        if (transcript.isNotEmpty) {
          print("GD: Turn transcribed: $transcript");
          setState(() {
            _sessionBotContext += "You: $transcript\n";
          });
        }
      } else {
        print("GD: Audio turn upload failed with status ${response.statusCode}");
      }
    } catch (e) {
      print("GD: Audio turn upload error: $e");
    }
  }

  // ── 4. FINISH & EVALUATE ─────────────────────────────────────────────────

  Future<void> _finishAndEvaluate() async {
    if (_currentState == GDState.evaluating) return;
    setState(() => _currentState = GDState.evaluating);

    _frameTimer?.cancel();   // Stop live frame warnings
    _sessionTimer?.cancel();

    SchedulerBinding.instance.addPostFrameCallback((_) {
      _tts.stop();
    });

    // Close WebSocket gracefully
    _wsConnected = false;
    try {
      await _wsChannel?.sink.close();
    } catch (_) {}

    String? videoPath = _currentVideoPath;

    try {
      // Catch any ongoing turn recording
      if (await _audioRecorder.isRecording()) {
        final path = await _audioRecorder.stop();
        if (path != null) _audioPaths.add(path);
      }

      // Stop video recording safely
      try {
        if (_cameraController != null && _cameraController!.value.isRecordingVideo) {
          final xfile = await _cameraController!.stopVideoRecording();
          videoPath = xfile.path;
          print("📸 GD Video Recording Stopped: $videoPath");
        } else {
          print("⚠️ NO video recording was active to stop.");
        }
      } catch (e) {
        print("Video stop error: $e");
      }

      final safeVideoPath = videoPath ?? "";
      if (safeVideoPath.isEmpty || !File(safeVideoPath).existsSync()) {
        throw Exception("Video recording missing or invalid. Session cannot be evaluated without camera data.");
      }

      print("🚀 GD: Submitting to backend (Topic: ${topic!['topic_id']}, Audio: ${_audioPaths.length} turns, Video: $safeVideoPath)");
      
      final evalResult = await ApiConfig.submitGD(
        topicId: topic!["topic_id"].toString(),
        audioFiles: _audioPaths.map((p) => File(p)).toList(),
        videoFile: File(safeVideoPath),
        botContext: _sessionBotContext,
      );

      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => ResultPage(result: evalResult)),
        );
      }
    } catch (e) {
      print("Evaluation error: $e");
      if (mounted) setState(() => _currentState = GDState.discussion);
    }
  }

  // ── 5. CAMERA & VIDEO ────────────────────────────────────────────────────

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      final front = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      _cameraController = CameraController(front, ResolutionPreset.medium);
      await _cameraController!.initialize();
      if (mounted) {
        setState(() {
          _cameraInitialized = true;
          _liveWarning = null; // Will be set by first frame analysis
        });
        _startLiveFrameAnalysis(); // Begin warnings immediately on instructions page
      }
    } catch (e) {
      print("Camera init error: $e");
    }
  }

  Future<void> _startVideoRecording() async {
    if (!_cameraInitialized) return;
    try {
      final dir = await getTemporaryDirectory();
      String newPath = "${dir.path}/video_${DateTime.now().millisecondsSinceEpoch}.mp4";
      await _cameraController!.startVideoRecording();
      setState(() {
        _currentVideoPath = newPath;
      });
      print("📸 GD Video Recording Started: $newPath");
    } catch (e) {
      print("Video recording start error: $e");
      setState(() {
        _currentVideoPath = null;
      });
    }
  }

  // ── 6. TOPIC FETCH ───────────────────────────────────────────────────────

  Future<void> fetchTopic() async {
    final data = await ApiConfig.fetchGDTopic();
    if (mounted) setState(() => topic = data);
  }

  // ── 7. LIFECYCLE ─────────────────────────────────────────────────────────

  @override
  void dispose() {
    _isDisposed = true;
    _wsConnected = false;
    _wsChannel?.sink.close();
    _tts.stop();
    _cameraController?.dispose();
    _audioRecorder.dispose();
    _timer?.cancel();
    _sessionTimer?.cancel();
    _frameTimer?.cancel();
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  // ── 8. BUILD ─────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F0C29),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (topic == null) {
      return const Center(child: CircularProgressIndicator(color: Colors.cyanAccent));
    }
    switch (_currentState) {
      case GDState.instructions:
        return _buildInstructionsView();
      case GDState.discussion:
        return _buildDiscussionView();
      case GDState.evaluating:
        return _buildEvaluatingView();
    }
  }

  Widget _buildEvaluatingView() {
    return const PremiumLoadingOverlay(message: "Analyzing your GD performance...");
  }

  // ── INSTRUCTIONS VIEW ─────────────────────────────────────────────────────

  Widget _buildInstructionsView() {
    final bool cameraLoading = !_cameraInitialized;
    final hasWarning = cameraLoading || _screenWarning ||
        (_liveWarning != null &&
            _liveWarning != 'LOOKING_GREAT' &&
            _liveWarning!.isNotEmpty);

    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [

            // ── Header ─────────────────────────────────────────────
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                // Back button
                GestureDetector(
                  onTap: () => Navigator.pop(context),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.06),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: Colors.white.withOpacity(0.12)),
                    ),
                    child: Row(
                      children: [
                        const Icon(Icons.arrow_back_ios_new, color: Colors.white70, size: 12),
                        const SizedBox(width: 5),
                        Text("Back",
                            style: GoogleFonts.inter(
                                color: Colors.white70, fontSize: 12, fontWeight: FontWeight.w500)),
                      ],
                    ),
                  ),
                ),
                // GROUP DISCUSSION badge
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                  decoration: BoxDecoration(
                    color: Colors.cyanAccent.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: Colors.cyanAccent.withOpacity(0.3)),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.groups_rounded, color: Colors.cyanAccent, size: 14),
                      const SizedBox(width: 6),
                      Text("GROUP DISCUSSION",
                          style: GoogleFonts.inter(
                              color: Colors.cyanAccent, fontSize: 11, fontWeight: FontWeight.bold, letterSpacing: 1.2)),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 14),
            Text("Pre-Session\nChecklist",
                style: GoogleFonts.inter(
                    fontSize: 28, fontWeight: FontWeight.bold, color: Colors.white, height: 1.2)),
            const SizedBox(height: 6),
            Text("Complete all checks before starting your GD session.",
                style: GoogleFonts.inter(fontSize: 13, color: Colors.white38)),

            const SizedBox(height: 20),

            // ── Topic card ─────────────────────────────────────────
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [Colors.cyanAccent.withOpacity(0.12), Colors.purpleAccent.withOpacity(0.08)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.cyanAccent.withOpacity(0.25)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("TODAY'S TOPIC",
                      style: GoogleFonts.inter(
                          fontSize: 10, fontWeight: FontWeight.bold, color: Colors.cyanAccent, letterSpacing: 1.5)),
                  const SizedBox(height: 6),
                  Text(topic!['topic'],
                      style: GoogleFonts.inter(fontSize: 15, fontWeight: FontWeight.w600, color: Colors.white, height: 1.4)),
                ],
              ),
            ),

            const SizedBox(height: 20),

            // ── Live camera preview ────────────────────────────────
            Container(
              height: 230,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.black26,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: hasWarning ? _getWarningColor() : Colors.white12,
                  width: hasWarning ? 2 : 1,
                ),
                boxShadow: [
                  if (!hasWarning)
                    BoxShadow(color: Colors.cyanAccent.withOpacity(0.08), blurRadius: 20),
                ],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(19),
                child: _cameraInitialized
                    ? _buildUserCameraWithWarning()
                    : const Center(child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.videocam_off, color: Colors.white24, size: 36),
                          SizedBox(height: 8),
                          Text("Initializing camera...", style: TextStyle(color: Colors.white24, fontSize: 12)),
                        ],
                      )),
              ),
            ),

            // Pre-check status pill
            const SizedBox(height: 10),
            Center(
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 400),
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                decoration: BoxDecoration(
                  color: hasWarning
                      ? _getWarningColor().withOpacity(0.15)
                      : Colors.greenAccent.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: hasWarning ? _getWarningColor().withOpacity(0.5) : Colors.greenAccent.withOpacity(0.4),
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      hasWarning ? Icons.warning_amber_rounded : Icons.check_circle_rounded,
                      size: 14,
                      color: hasWarning ? _getWarningColor() : Colors.greenAccent,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      hasWarning ? _getWarningMessage() : "Camera check passed",
                      style: GoogleFonts.inter(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: hasWarning ? _getWarningColor() : Colors.greenAccent,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 22),

            // ── Start button ───────────────────────────────────────
            SizedBox(
              width: double.infinity,
              height: 58,
              child: ElevatedButton(
                onPressed: hasWarning ? null : _startDiscussion,
                style: ElevatedButton.styleFrom(
                  padding: EdgeInsets.zero,
                  backgroundColor: Colors.transparent,
                  shadowColor: Colors.transparent,
                  disabledBackgroundColor: Colors.white10,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                ),
                child: Ink(
                  decoration: BoxDecoration(
                    gradient: hasWarning
                        ? null
                        : const LinearGradient(
                            colors: [Color(0xFF00E5FF), Color(0xFF7B2FBE)],
                            begin: Alignment.centerLeft,
                            end: Alignment.centerRight,
                          ),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Container(
                    alignment: Alignment.center,
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          hasWarning ? Icons.block : Icons.play_arrow_rounded,
                          color: hasWarning ? Colors.white30 : Colors.white,
                          size: 22,
                        ),
                        const SizedBox(width: 10),
                        Text(
                          hasWarning ? "FIX CAMERA TO START" : "BEGIN DISCUSSION",
                          style: GoogleFonts.inter(
                            color: hasWarning ? Colors.white30 : Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 15,
                            letterSpacing: 0.5,
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
    );
  }

  Widget _gdInstructionStep(String num, IconData icon, String title, String desc, Color color) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: const Color(0xFF161625),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: color.withOpacity(0.15)),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: color.withOpacity(0.12),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, color: color, size: 20),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(num,
                          style: GoogleFonts.jetBrainsMono(
                              fontSize: 10, color: color.withOpacity(0.5), fontWeight: FontWeight.bold)),
                      const SizedBox(width: 6),
                      Text(title,
                          style: GoogleFonts.inter(
                              fontSize: 14, fontWeight: FontWeight.bold, color: Colors.white)),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(desc,
                      style: GoogleFonts.inter(fontSize: 12, color: Colors.white54, height: 1.4)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }


  // ── DISCUSSION VIEW ───────────────────────────────────────────────────────

  Widget _buildDiscussionView() {
    return SafeArea(
      child: Column(
        children: [
          // Enhanced Header: Topic + Status Badges
          Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(20, 10, 20, 20),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [Colors.cyanAccent.withOpacity(0.08), Colors.transparent],
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
              ),
            ),
            child: Column(
              children: [
                // Status Row: Back + LIVE ... Timer + REC
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    // Left side: Exit button + "LIVE SESSION"
                    Row(
                      children: [
                        IconButton(
                          padding: EdgeInsets.zero,
                          constraints: const BoxConstraints(),
                          icon: const Icon(Icons.arrow_back_ios_new, color: Colors.white54, size: 14),
                          onPressed: () {
                            showDialog(
                              context: context,
                              builder: (context) => AlertDialog(
                                backgroundColor: const Color(0xFF161625),
                                title: const Text("Exit Session?", style: TextStyle(color: Colors.white)),
                                content: const Text("Your progress will be lost. Are you sure?", style: TextStyle(color: Colors.white70)),
                                actions: [
                                  TextButton(onPressed: () => Navigator.pop(context), child: const Text("CANCEL")),
                                  TextButton(
                                    onPressed: () {
                                      Navigator.pop(context);
                                      Navigator.pop(context);
                                    },
                                    child: const Text("EXIT", style: TextStyle(color: Colors.redAccent)),
                                  ),
                                ],
                              ),
                            );
                          },
                        ),
                        const SizedBox(width: 8),
                        Text(
                          "LIVE SESSION",
                          style: GoogleFonts.inter(
                            color: Colors.white54,
                            fontSize: 10,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 1.2,
                          ),
                        ),
                      ],
                    ),
                    // Right side: Timer + REC info
                    Row(
                      children: [
                        Text(
                          _formatDuration(_sessionCountdown),
                          style: GoogleFonts.jetBrainsMono(
                            color: _sessionCountdown <= 10 ? Colors.redAccent : Colors.cyanAccent,
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(width: 12),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: Colors.redAccent.withOpacity(0.12),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: Colors.redAccent.withOpacity(0.3)),
                          ),
                          child: Row(
                            children: [
                              const Icon(Icons.circle, color: Colors.redAccent, size: 6),
                              const SizedBox(width: 6),
                              Text(
                                "REC",
                                style: GoogleFonts.inter(
                                  color: Colors.redAccent,
                                  fontSize: 10,
                                  fontWeight: FontWeight.w900,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                // Heading Topic (Large & Centered)
                Text(
                  topic!["topic"].toString().toUpperCase(),
                  style: GoogleFonts.inter(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.w900,
                    letterSpacing: 0.2,
                    height: 1.2,
                  ),
                  textAlign: TextAlign.center,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),

          // Participant grid — 2 × 2
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12.0),
              child: Column(
                children: [
                  Expanded(
                    child: Row(
                      children: [
                        Expanded(
                          child: _participantTile(
                            "You",
                            _cameraInitialized
                                ? _buildUserCameraWithWarning()
                                : Container(color: Colors.black26),
                            _activeSpeaker == "User",
                            isWaiting: _handRaised,
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: _participantTile(
                            "Thomas",
                            const Icon(Icons.account_circle,
                                size: 50, color: Colors.amberAccent),
                            _activeSpeaker == "Bot_Mod",
                            isWaiting: _waitingSpeaker == "Bot_Mod",
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 10),
                  Expanded(
                    child: Row(
                      children: [
                        Expanded(
                          child: _participantTile(
                            "Aravind",
                            const Icon(Icons.account_circle,
                                size: 50, color: Colors.cyanAccent),
                            _activeSpeaker == "Bot_A",
                            isWaiting: _waitingSpeaker == "Bot_A",
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: _participantTile(
                            "George",
                            const Icon(Icons.account_circle,
                                size: 50, color: Colors.pinkAccent),
                            _activeSpeaker == "Bot_B",
                            isWaiting: _waitingSpeaker == "Bot_B",
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),



          _buildWarningBar(),
          _buildControlPanel(),
        ],
      ),
    );
  }

  // ── WIDGETS ───────────────────────────────────────────────────────────────

  /// Camera preview for "You" tile with live warning overlay
  Widget _buildUserCameraWithWarning() {
    final warningMsg = _getWarningMessage();
    final warningColor = _getWarningColor();
    final hasWarning = _screenWarning ||
        (_liveWarning != null &&
            _liveWarning != 'LOOKING_GREAT' &&
            _liveWarning!.isNotEmpty);

    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(_cameraController!),
        // Warning overlay banner
        if (warningMsg.isNotEmpty)
          Positioned(
            top: 8,
            left: 6,
            right: 6,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                child: Container(
                  padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 8),
                  color: hasWarning
                      ? warningColor.withOpacity(0.85)
                      : Colors.greenAccent.withOpacity(0.7),
                  child: Text(
                    warningMsg,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 10,
                    ),
                  ),
                ),
              ),
            ),
          ),
      ],
    );
  }

  Widget _participantTile(String name, Widget content, bool isSpeaking, {bool isWaiting = false}) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      decoration: BoxDecoration(
        color: const Color(0xFF161625),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: isSpeaking ? Colors.greenAccent : (isWaiting ? Colors.orangeAccent : Colors.white10),
          width: (isSpeaking || isWaiting) ? 2 : 1.0,
        ),
      ),
      child: Stack(
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(18),
            child: Center(child: content),
          ),
          Positioned(
            bottom: 10,
            left: 10,
            child: Text(
              name,
              style: const TextStyle(
                  color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold),
            ),
          ),
          if (isSpeaking)
            const Positioned(
              top: 10,
              right: 10,
              child: Icon(Icons.volume_up, color: Colors.greenAccent, size: 20),
            ),
          if (isWaiting && !isSpeaking)
            Positioned(
              top: 10,
              right: 10,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.orangeAccent.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.orangeAccent),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.pan_tool, color: Colors.orangeAccent, size: 14),
                    SizedBox(width: 4),
                    Text(
                      "Raised Hand",
                      style: TextStyle(
                        color: Colors.orangeAccent,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildControlPanel() {
    return Container(
      padding: const EdgeInsets.only(top: 15, bottom: 25, left: 20, right: 20),
      decoration: const BoxDecoration(
        color: Color(0xFF161625),
        borderRadius: BorderRadius.vertical(top: Radius.circular(30)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          // End session button
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              IconButton(
                icon: const Icon(Icons.call_end, color: Colors.redAccent, size: 28),
                onPressed: () {
                  showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                      backgroundColor: const Color(0xFF161625),
                      title: const Text("End Discussion?", style: TextStyle(color: Colors.white)),
                      content: const Text("Are you done? We will now evaluate your performance.", style: TextStyle(color: Colors.white70)),
                      actions: [
                        TextButton(onPressed: () => Navigator.pop(context), child: const Text("CANCEL")),
                        TextButton(
                          onPressed: () {
                            Navigator.pop(context); // Close dialog
                            _finishAndEvaluate();
                          },
                          child: const Text("END & EVALUATE", style: TextStyle(color: Colors.cyanAccent)),
                        ),
                      ],
                    ),
                  );
                },
                tooltip: "End Session",
              ),
              Text(
                "END",
                style: GoogleFonts.inter(
                    color: Colors.redAccent, fontSize: 9, fontWeight: FontWeight.bold),
              ),
            ],
          ),

          // Hand-raise / mic button
          GestureDetector(
            onTap: _toggleHand,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                CircleAvatar(
                  radius: 28,
                  backgroundColor: _recording
                      ? Colors.redAccent
                      : (_handRaised ? Colors.orangeAccent : Colors.cyanAccent),
                  child: Icon(
                    _recording ? Icons.mic : Icons.front_hand,
                    color: Colors.black,
                    size: 24,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  _recording
                      ? "LIVE"
                      : (_handRaised ? "WAIT..." : "RAISE HAND"),
                  style: GoogleFonts.inter(
                    color: _handRaised ? Colors.orangeAccent : Colors.white70,
                    fontSize: 9,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),

          const Icon(Icons.more_vert, color: Colors.white24),
        ],
      ),
    );
  }

  Widget _instructionTile(IconData icon, String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(children: [
        Icon(icon, color: Colors.cyanAccent, size: 20),
        const SizedBox(width: 15),
        Expanded(
          child: Text(text, style: const TextStyle(color: Colors.white70, fontSize: 14)),
        ),
      ]),
    );
  }

  Widget _buildWarningBar() {
    final msg = _getWarningMessage();
    final color = _getWarningColor();
    if (msg.isEmpty) return const SizedBox.shrink();

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 16),
      color: color.withOpacity(0.12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            _liveWarning == "LOOKING_GREAT" ? Icons.check_circle : Icons.warning_amber_rounded,
            color: color,
            size: 16,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              msg,
              style: GoogleFonts.inter(
                color: color,
                fontSize: 12,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }



  String _formatDuration(int seconds) {
    final min = (seconds ~/ 60).toString().padLeft(2, '0');
    final sec = (seconds % 60).toString().padLeft(2, '0');
    return "$min:$sec";
  }
}
