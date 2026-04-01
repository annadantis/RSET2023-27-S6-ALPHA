import json
import re
import ollama
import librosa
import numpy as np
import os

# Import your CameraEvaluator from camera_eval.py
from camera_eval import analyze_camera

# ---------------------------
# SILENCE & UTILITY
# ---------------------------

def is_silent_audio(audio_path: str) -> bool:
    """Check if audio file contains meaningful speech"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        intervals = librosa.effects.split(y, top_db=25)
        speech_time = sum((end - start) / sr for start, end in intervals)
        return speech_time < 1.0
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return True

def is_silent_transcript(transcript: str) -> bool:
    if not transcript: return True
    words = transcript.strip().split()
    return len(words) < 3

# ---------------------------
# OLLAMA RUNNER & JSON
# ---------------------------

def run_ollama(prompt: str) -> str:
    try:
        # Using format='json' for Llama3 reliability
        response = ollama.generate(model='llama3', prompt=prompt, format='json')
        return response['response'].strip()
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        raise

def extract_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON found in model response.")
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        fixed = match.group().replace("'", '"')
        return json.loads(fixed)

# ---------------------------
# MAIN EVALUATION FUNCTION
# ---------------------------

def evaluate_gd(topic: str, transcript: str, audio_path: str, video_path: str,
                target_keywords: list = None, bot_context: str = "") -> dict:
    """
    Full-spectrum GD Evaluation:
    1. Visual: Eye contact, Visibility, Smiles (via camera_eval.py)
    2. Audio: WPM, Fillers, Silence (via librosa)
    3. Contextual: Interaction with bots and Keyword coverage (via Llama3)
    """
    if target_keywords is None: target_keywords = []

    # --- 1. CAMERA ANALYSIS ---
    camera_results = analyze_camera(video_path)
    camera_score = camera_results.get("camera_score", 0.0)
    camera_feedback = camera_results.get("camera_feedback", "")
    camera_metrics = camera_results.get("metrics", {})

    # --- 2. HARD FAIL CHECK (Silence or No Face) ---
    if is_silent_audio(audio_path) or is_silent_transcript(transcript) or camera_score < 1.0:
        return {
            "overall_score": 0,
            "content_score": 0,
            "communication_score": 0,
            "voice_score": 0,
            "camera_score": camera_score,
            "feedback": f"Evaluation Failed: {camera_feedback if camera_score < 1.0 else 'No speech detected.'}",
            "improved_answer": "N/A",
            "ideal_answer": f"Ensure you are visible and speaking on: {topic}",
            "strategy_note": "Position yourself in a quiet, well-lit environment.",
            "found_keywords": [],
            "missing_keywords": target_keywords,
            "content_audit": {"error": "Invalid input data"},
            "camera_metrics": camera_metrics
        }

    # --- 3. VOICE ANALYSIS (Audio Metrics) ---
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        word_count = len(transcript.split())
        wpm = (word_count / duration) * 60 if duration > 0 else 0
        wpm_score = max(0, 10 - (abs(150 - wpm) / 10)) # Target 150 WPM
       
        intervals = librosa.effects.split(y, top_db=30)
        speech_time = sum((end - start) / sr for start, end in intervals)
        silence_pct = ((duration - speech_time) / duration) * 100
        silence_score = 10 if 10 <= silence_pct <= 20 else max(0, 10 - abs(15 - silence_pct) / 2)
       
        fillers = [r'\bum\b', r'\buh\b', r'\blike\b', r'\bactually\b', r'\bbasically\b', r'\byou know\b']
        filler_count = sum(len(re.findall(f, transcript.lower())) for f in fillers)
        filler_score = max(0, 10 - (filler_count * 1.5))

        final_voice_score = round(float((wpm_score * 0.4) + (silence_score * 0.3) + (filler_score * 0.3)), 1)
    except:
        final_voice_score, wpm, filler_count = 5.0, 0, 0

    # --- 4. KEYWORD & INTERACTION CHECK ---
    found_keywords = [k for k in target_keywords if k.lower() in transcript.lower()]
    missing_keywords = [k for k in target_keywords if k.lower() not in transcript.lower()]
   
    # Bonus for acknowledging bots (Aravind/George/Sneha)
    interaction_bonus = 0.0
    if re.search(r"(aravind|george|sneha|mentioned|point|agree|disagree|building|however)", transcript.lower()):
        interaction_bonus = 1.5

    # --- 5. OLLAMA ANALYSIS (Interaction & Behavioral) ---
    prompt = f"""
    You are a Senior Corporate HR Evaluator. Provide a strict analysis of this GD performance.
   
    TOPIC: "{topic}"
    BOT CONTEXT: "{bot_context}"
    USER TRANSCRIPT: "{transcript}"

    TECHNICAL METRICS:
    - Eye Contact: {camera_metrics.get('eye_contact', 0)}%
    - Voice Score: {final_voice_score}/10
    - Keywords Found: {found_keywords}

    EVALUATION TASKS:
    1. CONTENT: Logic, relevance, and use of technical keywords.
    2. COMMUNICATION: Did they respond to the bots effectively?
    3. BEHAVIORAL: Incorporate the eye contact/posture data into your feedback.

    MANDATORY JSON OUTPUT:
    {{
      "content_score": <0-10>,
      "communication_score": <0-10>,
      "content_audit": {{ "relevance": <0-10>, "depth": <0-10>, "structure": <0-10>, "vocabulary": <0-10> }},
      "feedback": "Be blunt and professional. Mention both speech content and eye contact behavior in 200 words",
      "improved_answer": "Professional rewrite using {target_keywords}.",
      "ideal_answer": "A comprehensive 200-word master response covering all major points of the topic and incorporating key technical terms.",
      "strategy_note": "One sentence on handling the transition from bot to user."
    }}
    """

    try:
        raw_output = run_ollama(prompt)
        result = extract_json(raw_output)
    except Exception as e:
        return {"error": "AI Processing Failed", "details": str(e)}

    # --- 6. FINAL SCORING & COMPILATION ---
    c_score = float(result.get("content_score", 0))
    # Blend: 60% LLM communication score + 40% voice quality (WPM, silence, fillers)
    comm_score = min(10.0, float(result.get("communication_score", 0)) * 0.6 + final_voice_score * 0.4 + interaction_bonus)

    # Weighted Score: 45% Content, 30% Communication (incl. Voice), 25% Camera
    overall_score = round(float((c_score * 0.45) + (comm_score * 0.30) + (camera_score * 0.25)), 1)

    # Final mapping
    result.update({
        "overall_score": overall_score,
        "content_score": c_score,
        "communication_score": comm_score,
        "voice_score": final_voice_score,
        "camera_score": camera_score,
        "camera_feedback": camera_feedback,
        "camera_metrics": camera_metrics,
        "found_keywords": found_keywords,
        "missing_keywords": missing_keywords,
        "transcript": transcript,
        "wpm": round(wpm, 1),
        "filler_count": filler_count
    })

    return result

# ---------------------------
# TEST BLOCK
# ---------------------------
if __name__ == "__main__":
    # Ensure test files exist before running this
    print("Executing full GD evaluation test...")
    # Example call:
    # report = evaluate_gd(topic="AI Ethics", transcript="...", audio_path="a.wav", video_path="v.mp4")



