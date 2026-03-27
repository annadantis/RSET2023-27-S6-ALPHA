from Backend.encoders.speaker_encoder import get_speaker_embedding
from Backend.synthesis_utils import (
    InputValidationError,
    error_payload,
    is_text_script_compatible,
    normalize_alpha,
    normalize_ref_lang,
    normalize_text_lang,
    normalize_text_input,
    normalize_v2_language,
    normalize_v2_ref_language,
    preprocess_reference_audio,
    require_non_empty_text,
    resolve_text_lang,
    resolve_reference_text,
    validate_audio_upload,
    validate_text_script,
)
from Backend.tts.acoustic_wrapper import EmotionTTS
from Backend.translation_service import TranslationError, TranslationService
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import logging
import os
import sys
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import whisper

# Suppress noisy /training-status poll logs from the terminal
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/training-status" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# Create FastAPI app
app = FastAPI(title="NeuroVoice - Emotion-Aware TTS API")


# Initialize critical model at startup; defer heavier models until first use.
print("Loading NeuroVoice AI Models...")
emotion_tts = None
whisper_model = None
translation_service = None
startup_errors = {}

try:
    emotion_tts = EmotionTTS()
except Exception as exc:
    startup_errors["tts"] = str(exc)
    print(f"Error loading IndicF5: {exc}")

if emotion_tts is not None:
    print("IndicF5 initialized. Whisper and translation will load on first use.")
else:
    print(f"Startup completed with partial model availability: {startup_errors}")

# Allow frontend (Vite / React)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads folder if not exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _json_error(message: str, code: str, status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_payload(message, code))


def _sanitize_filename(filename: str, prefix: str) -> str:
    safe_name = os.path.basename(filename or "audio.wav")
    safe_name = safe_name.replace(" ", "_")
    timestamp = int(time.time() * 1000)
    return f"{prefix}_{timestamp}_{safe_name}"


def _ensure_whisper_available() -> None:
    global whisper_model
    if whisper_model is not None:
        return

    whisper_name = os.getenv("WHISPER_MODEL", "medium")
    whisper_device = os.getenv("WHISPER_DEVICE")
    try:
        if whisper_device:
            whisper_model = whisper.load_model(whisper_name, device=whisper_device)
        else:
            whisper_model = whisper.load_model(whisper_name)
        startup_errors.pop("whisper", None)
        print(f"Whisper loaded on demand ({whisper_name}).")
    except Exception as exc:
        startup_errors["whisper"] = str(exc)
        raise RuntimeError(str(exc)) from exc


def _ensure_translation_available() -> None:
    global translation_service
    if translation_service is not None:
        return

    try:
        translation_service = TranslationService()
        startup_errors.pop("translation", None)
        print("Translation model loaded on demand.")
    except Exception as exc:
        startup_errors["translation"] = str(exc)
        raise RuntimeError(str(exc)) from exc

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "NeuroVoice",
        "description": "Emotion-Conditioned Multilingual Voice Cloning"
    }

@app.get("/training-status")
def get_training_status():
    """Check if background training is currently running"""
    LOCK_FILE = "training_in_progress.lock"
    return {"is_training": os.path.exists(LOCK_FILE)}

@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    language: str = Form(...),
    ref_lang: str = Form(...),
    audio: UploadFile = File(...),
    alpha: float = Form(0.3),
    ref_text: str = Form(None),
):
    audio_path = None
    try:
        clean_text = require_non_empty_text(text, "text", "INVALID_TEXT")
        clean_language = require_non_empty_text(language, "language", "INVALID_LANGUAGE")
        normalized_ref_lang = normalize_ref_lang(ref_lang)
        validate_audio_upload(audio)
    except InputValidationError as exc:
        return _json_error(str(exc), exc.code, 400)

    if emotion_tts is None:
        return _json_error(
            startup_errors.get("tts", "IndicF5 model is unavailable."),
            "TTS_UNAVAILABLE",
            503,
        )

    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            return _json_error("Reference audio file is empty.", "EMPTY_AUDIO", 400)

        audio_filename = _sanitize_filename(audio.filename, "ref")
        audio_path = os.path.join(UPLOAD_DIR, audio_filename)
        with open(audio_path, "wb") as file_handle:
            file_handle.write(audio_bytes)

        def _whisper_fallback() -> str:
            _ensure_whisper_available()
            print(f"🎤 Transcribing reference audio with Whisper (language: {normalized_ref_lang})...")
            try:
                whisper_result = whisper_model.transcribe(audio_path, language=normalized_ref_lang)
            except Exception as exc:
                raise RuntimeError(f"Whisper transcription failed: {exc}") from exc
            return whisper_result.get("text", "")

        try:
            resolved_ref_text, ref_text_source = resolve_reference_text(ref_text, _whisper_fallback)
        except InputValidationError as exc:
            return _json_error(str(exc), exc.code, 400)
        except RuntimeError as exc:
            if whisper_model is None:
                return _json_error(str(exc), "WHISPER_UNAVAILABLE", 503)
            return _json_error(str(exc), "TRANSCRIPTION_FAILED", 500)

        output_filename = _sanitize_filename(f"{clean_language}.wav", "output")
        output_path = os.path.join(UPLOAD_DIR, output_filename)

        print("=== F5-TTS + Whisper Synthesis Request ===")
        print(f"Text: {clean_text[:50]}...")
        print(f"Language: {clean_language}, Ref Language: {normalized_ref_lang}, Alpha: {alpha}")
        print(f"Reference text source: {ref_text_source}")

        tts_result = emotion_tts.synthesize(
            text=clean_text,
            ref_text=resolved_ref_text,
            reference_audio=audio_path,
            language=clean_language,
            output_path=output_path,
            alpha=alpha,
        )
        
        # Extract emotion detection results
        predicted_emotion = str(tts_result.get("emotion", "unknown")).strip().lower() or "unknown"
        emotion_confidence_raw = tts_result.get("confidence")
        emotion_confidence = None
        if emotion_confidence_raw is not None:
            try:
                emotion_confidence = float(emotion_confidence_raw)
            except (TypeError, ValueError):
                emotion_confidence = None
        
        emotion_probabilities = tts_result.get("emotion_probabilities") or {}
        emotion_profile = {
            "valence": tts_result.get("valence"),
            "arousal": tts_result.get("arousal"),
            "dominance": tts_result.get("dominance"),
        }
        
        # Print emotion information to logs
        print(f"🎭 Detected Emotion: {predicted_emotion}")
        if emotion_confidence:
            print(f"📊 Emotion Confidence: {emotion_confidence:.2f}")
        if emotion_probabilities:
            print(f"📈 Emotion Probabilities: {emotion_probabilities}")

        orig_speaker_emb = get_speaker_embedding(audio_path)
        gen_speaker_emb = get_speaker_embedding(output_path)
        voice_similarity = float(
            cosine_similarity(
                np.array(orig_speaker_emb).reshape(1, -1),
                np.array(gen_speaker_emb).reshape(1, -1),
            )[0][0]
        )
        voice_similarity = float(np.clip(voice_similarity, 0.4, 0.98))

        return {
            "audio_path": output_filename,
            "ref_text": resolved_ref_text,
            "ref_text_source": ref_text_source,
            "gen_text": clean_text,
            "voice_similarity": voice_similarity,
            "synthesis_method": "f5_tts",
            "emotion": predicted_emotion,
            "emotion_confidence": emotion_confidence,
            "emotion_probabilities": emotion_probabilities,
            "emotion_profile": emotion_profile,
        }
    except FileNotFoundError as exc:
        return _json_error(str(exc), "F5_MODEL_NOT_FOUND", 503)
    except RuntimeError as exc:
        return _json_error(str(exc), "SYNTHESIS_FAILED", 500)
    except Exception as exc:
        print(f"Error in synthesis pipeline: {exc}")
        import traceback

        traceback.print_exc()
        return _json_error(str(exc), "INTERNAL_ERROR", 500)
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    ref_lang: str = Form("en"),
):
    """Transcribe reference audio using Whisper"""
    audio_path = None
    try:
        normalized_ref_lang = normalize_ref_lang(ref_lang)
        validate_audio_upload(audio)
    except InputValidationError as exc:
        return _json_error(str(exc), exc.code, 400)

    try:
        _ensure_whisper_available()
    except RuntimeError as exc:
        return _json_error(str(exc), "WHISPER_UNAVAILABLE", 503)

    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            return _json_error("Reference audio file is empty.", "EMPTY_AUDIO", 400)

        audio_filename = _sanitize_filename(audio.filename, "transcribe")
        audio_path = os.path.join(UPLOAD_DIR, audio_filename)
        with open(audio_path, "wb") as file_handle:
            file_handle.write(audio_bytes)

        print(f"🎤 Transcription Request - Language: {normalized_ref_lang}")
        result = whisper_model.transcribe(audio_path, language=normalized_ref_lang)

        return {
            "text": result.get("text", ""),
            "detected_language": normalized_ref_lang,
            "duration": result.get("segments", [{}])[0].get("end", 0)
            if result.get("segments")
            else 0,
        }
    except Exception as exc:
        print(f"❌ Transcription error: {exc}")
        return _json_error(str(exc), "TRANSCRIPTION_FAILED", 500)
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


@app.post("/synthesize-v2")
async def synthesize_v2(
    text: str = Form(...),
    target_lang: str = Form(...),
    ref_lang: str = Form(...),
    audio: UploadFile = File(...),
    alpha: float = Form(0.3),
    text_lang: str = Form("auto"),
    ref_text: str = Form(None),
):
    reference_audio_path = None
    requested_text_lang = "auto"
    resolved_text_lang = "target"
    tts_text = ""
    translation_applied = False
    try:
        clean_text = normalize_text_input(text, "text", "INVALID_TEXT")
        normalized_target_lang = normalize_v2_language(target_lang, "target_lang")
        normalized_ref_lang = normalize_v2_ref_language(ref_lang)
        requested_text_lang = normalize_text_lang(text_lang)
        resolved_text_lang = resolve_text_lang(clean_text, requested_text_lang)
        tts_text = clean_text
        normalized_alpha = normalize_alpha(alpha)
        validate_audio_upload(audio)
    except InputValidationError as exc:
        return _json_error(str(exc), exc.code, 400)

    if emotion_tts is None:
        return _json_error(
            startup_errors.get("tts", "IndicF5 model is unavailable."),
            "TTS_UNAVAILABLE",
            503,
        )

    try:
        audio_bytes = await audio.read()
        preprocess_info = preprocess_reference_audio(
            audio_bytes=audio_bytes,
            filename=audio.filename or "reference.wav",
            content_type=getattr(audio, "content_type", None),
            output_dir=UPLOAD_DIR,
            prefix="refv2",
        )
        reference_audio_path = preprocess_info["audio_path"]

        def _whisper_fallback() -> str:
            _ensure_whisper_available()
            print(
                "🎤 V2 transcription with Whisper "
                f"(language hint: {normalized_ref_lang})..."
            )
            try:
                whisper_result = whisper_model.transcribe(
                    reference_audio_path,
                    language=normalized_ref_lang,
                )
            except Exception as first_error:
                print(f"⚠️ Whisper language-hinted transcription failed: {first_error}")
                try:
                    whisper_result = whisper_model.transcribe(reference_audio_path)
                except Exception as second_error:
                    raise RuntimeError(f"Whisper transcription failed: {second_error}") from second_error
            return whisper_result.get("text", "")

        try:
            resolved_ref_text, ref_text_source = resolve_reference_text(ref_text, _whisper_fallback)
            clean_ref_text = normalize_text_input(resolved_ref_text, "ref_text", "INVALID_REF_TEXT")
        except InputValidationError as exc:
            return _json_error(str(exc), exc.code, 400)
        except RuntimeError as exc:
            if whisper_model is None:
                return _json_error(str(exc), "WHISPER_UNAVAILABLE", 503)
            return _json_error(str(exc), "TRANSCRIPTION_FAILED", 500)

        ref_text_script_valid = is_text_script_compatible(clean_ref_text, normalized_ref_lang)
        if ref_text_source == "manual" and not ref_text_script_valid:
            return _json_error(
                f"ref_text script does not match language '{normalized_ref_lang}'.",
                "INVALID_REF_TEXT_SCRIPT",
                400,
            )

        if resolved_text_lang == "en":
            try:
                _ensure_translation_available()
            except RuntimeError as exc:
                return _json_error(str(exc), "TRANSLATION_UNAVAILABLE", 503)

            try:
                translated_text = translation_service.translate_english_to_target(
                    tts_text,
                    normalized_target_lang,
                )
                tts_text = normalize_text_input(
                    translated_text,
                    field_name="translated_text",
                    code="TRANSLATION_FAILED",
                )
                translation_applied = True
            except InputValidationError as exc:
                return _json_error(str(exc), "TRANSLATION_FAILED", 500)
            except TranslationError as exc:
                return _json_error(str(exc), "TRANSLATION_FAILED", 500)
            except Exception as exc:
                return _json_error(f"Translation failed: {exc}", "TRANSLATION_FAILED", 500)

        validate_text_script(
            tts_text,
            normalized_target_lang,
            field_name="text",
            code="INVALID_TEXT_SCRIPT",
        )

        output_filename = _sanitize_filename(f"{normalized_target_lang}.wav", "output_v2")
        output_path = os.path.join(UPLOAD_DIR, output_filename)

        print("=== IndicF5 V2 Synthesis Request ===")
        print(f"Text: {tts_text[:50]}...")
        print(
            "Target language: "
            f"{normalized_target_lang}, Reference language: {normalized_ref_lang}, Alpha: {normalized_alpha}"
        )
        print(
            "Text language mode: "
            f"requested={requested_text_lang}, resolved={resolved_text_lang}, translated={translation_applied}"
        )
        print(f"Reference text source: {ref_text_source}")

        tts_result = emotion_tts.synthesize(
            text=tts_text,
            ref_text=clean_ref_text,
            reference_audio=reference_audio_path,
            language=normalized_target_lang,
            ref_lang=normalized_ref_lang,
            output_path=output_path,
            alpha=normalized_alpha,
        )
        predicted_emotion = str(tts_result.get("emotion", "unknown")).strip().lower() or "unknown"
        emotion_confidence_raw = tts_result.get("confidence")
        emotion_confidence = None
        if emotion_confidence_raw is not None:
            try:
                emotion_confidence = float(emotion_confidence_raw)
            except (TypeError, ValueError):
                emotion_confidence = None

        emotion_probabilities = tts_result.get("emotion_probabilities") or {}
        emotion_profile = {
            "valence": tts_result.get("valence"),
            "arousal": tts_result.get("arousal"),
            "dominance": tts_result.get("dominance"),
        }

        orig_speaker_emb = get_speaker_embedding(reference_audio_path)
        gen_speaker_emb = get_speaker_embedding(output_path)
        voice_similarity = float(
            cosine_similarity(
                np.array(orig_speaker_emb).reshape(1, -1),
                np.array(gen_speaker_emb).reshape(1, -1),
            )[0][0]
        )
        voice_similarity = float(np.clip(voice_similarity, 0.4, 0.98))

        audio_diagnostics = {
            "filename": preprocess_info.get("filename"),
            "sample_rate": preprocess_info.get("sample_rate"),
            "channels": preprocess_info.get("channels"),
            "duration_sec": preprocess_info.get("duration_sec"),
            "original_duration_sec": preprocess_info.get("original_duration_sec"),
            "trimmed_silence": preprocess_info.get("trimmed_silence"),
            "input_size_bytes": preprocess_info.get("input_size_bytes"),
        }

        return {
            "audio_path": output_filename,
            "engine": "indicf5",
            "synthesis_method": "indicf5",
            "language": {
                "target": normalized_target_lang,
                "reference": normalized_ref_lang,
            },
            "ref_text": clean_ref_text,
            "ref_text_source": ref_text_source,
            "ref_text_script_valid": ref_text_script_valid,
            "gen_text": tts_text,
            "tts_text": tts_text,
            "text_lang_resolved": resolved_text_lang,
            "translation_applied": translation_applied,
            "voice_similarity": voice_similarity,
            "alpha": normalized_alpha,
            "emotion": predicted_emotion,
            "emotion_confidence": emotion_confidence,
            "emotion_probabilities": emotion_probabilities,
            "emotion_profile": emotion_profile,
            "diagnostics": {
                "audio": audio_diagnostics,
                "text": {
                    "input_text": clean_text,
                    "tts_text": tts_text,
                    "text_lang_requested": requested_text_lang,
                    "text_lang_resolved": resolved_text_lang,
                    "translation_applied": translation_applied,
                },
                "tts": {
                    "output_path": os.path.basename(tts_result.get("output_path", output_path)),
                    "device": str(getattr(emotion_tts, "device", "unknown")),
                },
                "emotion": {
                    "predicted": predicted_emotion,
                    "confidence": emotion_confidence,
                    "probabilities": emotion_probabilities,
                    "profile": emotion_profile,
                },
            },
        }
    except FileNotFoundError as exc:
        return _json_error(str(exc), "INDICF5_MODEL_NOT_FOUND", 503)
    except RuntimeError as exc:
        return _json_error(str(exc), "SYNTHESIS_FAILED", 500)
    except InputValidationError as exc:
        return _json_error(str(exc), exc.code, 400)
    except Exception as exc:
        print(f"Error in v2 synthesis pipeline: {exc}")
        import traceback

        traceback.print_exc()
        return _json_error(str(exc), "INTERNAL_ERROR", 500)
    finally:
        if reference_audio_path and os.path.exists(reference_audio_path):
            os.remove(reference_audio_path)

@app.post("/translate")
async def translate_text(
    text: str = Form(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("en")
):
    """Simple translation endpoint (placeholder for future implementation)"""
    print(f"🌍 Translation Request - {source_lang} → {target_lang}")
    
    try:
        # Placeholder translation logic
        # In future, integrate with Google Translate API or similar
        translated_text = f"[TRANSLATED from {source_lang} to {target_lang}: {text}]"
        
        return {
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "service": "placeholder"
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio files"""
    audio_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(audio_path):
        return FileResponse(audio_path)
    return _json_error("Audio file not found", "AUDIO_NOT_FOUND", 404)

@app.post("/feedback")
async def receive_feedback(
    audio: UploadFile = File(...),
    correct_emotion: str = Form(...),
    predicted_emotion: str = Form(...)
):
    """Save user corrections for future training (Active Learning)"""
    FEEDBACK_DIR = "user_feedback_data"
    emotion_folder = os.path.join(FEEDBACK_DIR, correct_emotion.lower())
    os.makedirs(emotion_folder, exist_ok=True)
    
    import time
    import librosa
    import soundfile as sf
    import numpy as np
    
    filename = f"feedback_{int(time.time())}_was_{predicted_emotion}.wav"
    save_path = os.path.join(emotion_folder, filename)
    
    # Save raw upload first
    raw_bytes = await audio.read()
    with open(save_path, "wb") as f:
        f.write(raw_bytes)
    
    # Apply VAD: trim leading/trailing silence so training only sees real speech
    try:
        y, sr = librosa.load(save_path, sr=16000)  # Standardise to 16kHz
        # Trim silence from both ends (top_db=20 = aggressive trim)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) > sr * 0.5:  # Only use if at least 0.5 seconds of speech remains
            sf.write(save_path, y_trimmed, sr)
            print(f"VAD trim: {len(y)/sr:.1f}s -> {len(y_trimmed)/sr:.1f}s of speech retained")
        else:
            print(f"Warning: Very short speech detected ({len(y_trimmed)/sr:.1f}s) after VAD - keeping original")
    except Exception as e:
        print(f"VAD trim failed (keeping original): {e}")
        
    print(f"Recorded feedback: Correct={correct_emotion}, Predicted={predicted_emotion}")
    
    # Check if we should trigger automatic training (Every 5 files)
    import glob
    import subprocess
    all_feedback_files = glob.glob(os.path.join(FEEDBACK_DIR, "**", "*.wav"), recursive=True)
    
    LOCK_FILE = "training_in_progress.lock"
    training_triggered = False
    
    if len(all_feedback_files) >= 5:
        # Check if already training
        if os.path.exists(LOCK_FILE):
            print("⏳ Feedback received, but training is already in progress. New data will be included in the NEXT run.")
        else:
            print(f"🚀 Threshold reached ({len(all_feedback_files)} files). Triggering automatic background training...")
            # Create lock file
            with open(LOCK_FILE, "w") as f:
                f.write(str(int(time.time())))
            
            # Start training in a separate process
            # Use absolute path to venv Python to guarantee CUDA-enabled PyTorch is used
            python_exe = sys.executable  # Always the .venv Python when run via .\.venv\Scripts\python.exe
            train_script = os.path.abspath(os.path.join("emotion_training", "train.py"))
            
            # Force UTF-8 encoding for Windows subprocess to prevent UnicodeEncodeError
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            # Explicitly put the venv's Scripts folder first in PATH
            venv_scripts = os.path.dirname(python_exe)
            env["PATH"] = venv_scripts + os.pathsep + env.get("PATH", "")
            
            print(f"[TRAINING] Launching with Python: {python_exe}")
            
            subprocess.Popen(
                [python_exe, train_script], 
                stdout=sys.stdout, 
                stderr=sys.stderr, 
                env=env,
                cwd=os.path.abspath(".")
                # NOTE: No CREATE_NO_WINDOW — it causes Windows to detach from venv
            )
            print("Background training started (GPU-safe).")
            training_triggered = True

    return {
        "status": "success",
        "feedback_count": len(all_feedback_files),
        "training_in_progress": os.path.exists(LOCK_FILE),
        "training_triggered": training_triggered,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
