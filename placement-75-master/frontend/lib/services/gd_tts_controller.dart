import 'package:flutter_tts/flutter_tts.dart';
import 'dart:io' show Platform;
import 'dart:async';

class GDTTSController {
  static final GDTTSController _instance = GDTTSController._internal();
  final FlutterTts _flutterTts = FlutterTts();
  final Completer<void> _initCompleter = Completer<void>();

  // Queue so messages are spoken one at a time
  final List<Map<String, String>> _queue = [];
  bool _isSpeaking = false;
  Function? onQueueEmpty;

  factory GDTTSController() => _instance;

  // Map to store specific voice configurations for each bot
  final Map<String, Map<String, String>> _botVoices = {};

  GDTTSController._internal() {
    _initTTS();
  }

  void _initTTS() async {
    try {
      final engines = await _flutterTts.getEngines;
      print("TTS: Available Engines: $engines");

      final languages = await _flutterTts.getLanguages;
      print("TTS: Available Languages: $languages");

      final voices = await _flutterTts.getVoices;
      print("TTS: Total voices found: ${voices.length}");

      // Try to set a good English language first
      bool langSet = false;
      String selectedLang = "en-US";
      for (final lang in ["en-IN", "en-GB", "en-US", "en"]) {
        if (languages.contains(lang) ||
            languages.contains(lang.replaceAll('-', '_'))) {
          try {
            await _flutterTts.setLanguage(lang);
            print("TTS: Successfully set language to $lang");
            selectedLang = lang;
            langSet = true;
            break;
          } catch (_) {}
        }
      }
      
      // Assign distinct voices to each bot if available
      try {
        List<Map<String, String>> enVoices = [];
        for (dynamic v in voices) {
          if (v is Map) {
            final voiceMap = Map<String, String>.from(v.map((key, value) => MapEntry(key.toString(), value.toString())));
            final locale = voiceMap['locale'] ?? '';
            if (locale.startsWith('en')) {
              enVoices.add(voiceMap);
            }
          }
        }

        print("TTS: Found ${enVoices.length} English voices.");

        if (enVoices.isNotEmpty) {
          // Thomas (Moderator)
          _botVoices['Bot_Mod'] = enVoices[0];
          
          // Aravind (Challenger)
          if (enVoices.length > 1) {
            _botVoices['Bot_A'] = enVoices[1];
          } else {
            _botVoices['Bot_A'] = enVoices[0];
          }

          // George (Supporter)
          if (enVoices.length > 2) {
            _botVoices['Bot_B'] = enVoices[2];
          } else if (enVoices.length > 1) {
            _botVoices['Bot_B'] = enVoices[1];
          } else {
            _botVoices['Bot_B'] = enVoices[0];
          }
          
          print("TTS: Assigned ${enVoices.length > 2 ? '3 distinct' : 'fallback'} voices to bots.");
        }
      } catch (e) {
        print("TTS Error during voice assignment: $e");
      }

      if (!langSet && languages.isNotEmpty) {
        await _flutterTts.setLanguage(languages.first);
        print("TTS: Fallback to first available: ${languages.first}");
      }

      try { await _flutterTts.setSpeechRate(0.65); } catch (_) {}
      try { await _flutterTts.setVolume(1.0);     } catch (_) {}

      // On Windows, awaitSpeakCompletion(true) can cause severe threading crashes
      try { await _flutterTts.awaitSpeakCompletion(false); } catch (_) {}

      if (!_initCompleter.isCompleted) _initCompleter.complete();
      print("TTS: Controller Initialized Successfully");
    } catch (e) {
      print("TTS Error: Initialization failed: $e");
      if (!_initCompleter.isCompleted) _initCompleter.complete();
    }
  }

  /// Adds a message to the queue and starts processing if idle
  Future<void> speak(String text, String speaker) async {
    _queue.add({'text': text, 'speaker': speaker});
    print("TTS: Queued message for '$speaker'. Queue size: ${_queue.length}");
    if (!_isSpeaking) {
      _processQueue();
    }
  }

  Future<void> _processQueue() async {
    if (_isSpeaking) return;
    _isSpeaking = true;

    await _initCompleter.future;

    while (_queue.isNotEmpty) {
      final next = _queue.removeAt(0);
      final text = next['text']!;
      final speaker = next['speaker']!;

      print("TTS: Bot '$speaker' is speaking: $text");

      try {
        // Apply individual voice if assigned
        if (_botVoices.containsKey(speaker)) {
          print("TTS: Setting voice for $speaker: ${_botVoices[speaker]!['name']}");
          await _flutterTts.setVoice(_botVoices[speaker]!);
        }

        // Keep pitch as secondary differentiator
        if (speaker == "Bot_Mod") {
          await _flutterTts.setPitch(1.0);
        } else if (speaker == "Bot_A") {
          await _flutterTts.setPitch(0.85);
        } else if (speaker == "Bot_B") {
          await _flutterTts.setPitch(0.8);
        }

        await _flutterTts.speak(text);

        // Word-count based delay
        final wordCount = text.trim().split(RegExp(r'\s+')).length;
        final delayMs = (wordCount * 462) + 900;
        await Future.delayed(Duration(milliseconds: delayMs));
        
      } catch (e) {
        print("TTS Error during speak: $e");
      }
    }
    
    _isSpeaking = false;
    onQueueEmpty?.call();
  }

  Future<void> stop() async {
    try {
      _queue.clear();
      _isSpeaking = false;
      await _flutterTts.stop();
    } catch (e) {
      print("TTS Error during stop: $e");
    }
  }
}



