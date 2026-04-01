import json
import re
import subprocess
import librosa
import ollama

# ---------------------------
# SILENCE DETECTION
# ---------------------------

def is_silent_audio(audio_path: str) -> bool:
    try:
        y, sr = librosa.load(audio_path, sr=None)
        intervals = librosa.effects.split(y, top_db=25)
        speech_time = sum((end - start) / sr for start, end in intervals)
        return speech_time < 1.0  # less than 1 sec speech = silent
    except Exception:
        return True


def is_silent_transcript(transcript: str) -> bool:
    if not transcript:
        return True
    words = transcript.strip().split()
    return len(words) < 3


# ---------------------------
# OLLAMA RUNNER
# ---------------------------

def run_ollama(prompt: str) -> str:
    """
    Runs Ollama using the official library for better JSON handling.
    """
    try:
        response = ollama.generate(
            model="llama3.2:1b",
            prompt=prompt,
            format="json",
            options={
                "temperature": 0.1,  # Low temperature for more consistent JSON
                "num_predict": 500   # Limit output length
            }
        )
        return response['response'].strip()
    except Exception as e:
        print(f"Ollama Library Error: {e}")
        # Fallback to empty JSON object if library fails
        return "{}"


# ---------------------------
# JSON EXTRACTION (ROBUST)
# ---------------------------

def extract_json(text: str) -> dict:
    """
    Robustly extract the FIRST valid JSON object from the text.
    """
    text = text.strip()
    
    # Try to find all possible JSON-like substrings
    # We find all { and try to find a matching }
    potential_json_objects = []
    
    # Simple regex to find content between { }
    # This won't handle nested objects perfectly with regex alone,
    # so we'll do a bit of manual scanning.
    
    start_indices = [m.start() for m in re.finditer(r'{', text)]
    
    for start in start_indices:
        # For each start, find the closing brace by counting
        count = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                count += 1
            elif text[i] == '}':
                count -= 1
                if count == 0:
                    # Found a potential JSON object
                    candidate = text[start:i+1]
                    try:
                        obj = json.loads(candidate)
                        return obj # Return the first valid one
                    except:
                        pass
                    break
    
    # If the above fails, fallback to the old method but try to be smarter
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except:
            pass

    raise ValueError(f"No valid JSON object found in output.\nRaw output: {text}")


# ---------------------------
# MAIN EVALUATION FUNCTION
# ---------------------------

def evaluate_gd(topic: str, transcript: str, audio_path: str, video_path: str) -> dict:
    """
    Central GD evaluation logic.
    ALL strictness and penalties handled here.
    """
    from camera_eval import analyze_camera

    # ---------------------------
    # HARD FAIL: SILENCE
    # ---------------------------
    if is_silent_audio(audio_path) or is_silent_transcript(transcript):
        return {
            "transcript": transcript or "",
            "content_score": 0,
            "communication_score": 0,
            "camera_score": 0,
            "final_score": 0,
            "feedback": "No meaningful speech detected. Please speak clearly and stay on topic.",
            "camera_feedback": "No face detected or no speech.",
            "ideal_answer": ""
        }

    # ---------------------------
    # OLLAMA PROMPT
    # ---------------------------
    prompt = f"""
You are an expert, strict English Group Discussion evaluator. 

Topic: "{topic}"
User Transcript: "{transcript}"

Instruction:
1. Evaluate the user's transcript based on the topic.
2. Provide feedback in ENGLISH only.
3. Provide a model 'ideal_answer' in ENGLISH only.
4. Return exactly ONE JSON object. 
5. Do NOT provide multiple options or extra text.

Return Format:
{{
  "content_score": <int 0-10>,
  "communication_score": <int 0-10>,
  "feedback": "Concise English feedback here",
  "ideal_answer": "Model English response here"
}}

Rules:
- Return ONLY the JSON object. 
- Be VERY strict.
- If the user is off-topic, give low scores.
- Language must be English.
"""

    raw_output = run_ollama(prompt)

    # ---------------------------
    # PARSE JSON
    # ---------------------------
    try:
        result = extract_json(raw_output)
    except Exception as e:
        raise RuntimeError(
            f"Ollama returned invalid JSON.\nRaw output:\n{raw_output}"
        ) from e

    # ---------------------------
    # NORMALIZE SCORES
    # ---------------------------
    result["content_score"] = int(max(0, min(10, result.get("content_score", 0))))
    result["communication_score"] = int(max(0, min(10, result.get("communication_score", 0))))

    # ---------------------------
    # STRICT PENALTIES
    # ---------------------------
    word_count = len(transcript.split())

    # Very short answer → heavy penalty
    if word_count < 10:
        result["content_score"] = min(result["content_score"], 3)
        result["communication_score"] = min(result["communication_score"], 3)

    # Weak communication → cap content
    if result["communication_score"] <= 2:
        result["content_score"] = min(result["content_score"], 4)

    # ---------------------------
    # CAMERA ANALYSIS
    # ---------------------------
    try:
        camera = analyze_camera(video_path)
        camera_score = int(max(0, min(10, camera.get("camera_score", 5))))
        camera_feedback = camera.get("camera_feedback", "")
    except Exception:
        camera_score = 5
        camera_feedback = "Camera analysis failed."

    # ---------------------------
    # FINAL SCORE CALCULATION
    # ---------------------------
    final_score = round(
        result["content_score"] * 0.5 +
        result["communication_score"] * 0.3 +
        camera_score * 0.2
    )

    # ---------------------------
    # FINAL RESPONSE
    # ---------------------------
    print("OLLAMA RESULT:", result)

    return {
        "transcript": transcript,
        "content_score": result["content_score"],
        "communication_score": result["communication_score"],
        "camera_score": camera_score,
        "final_score": final_score,
        "feedback": result["feedback"],
        "camera_feedback": camera_feedback,
        "ideal_answer": result["ideal_answer"]
    }
