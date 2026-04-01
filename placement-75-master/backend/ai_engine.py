import os
import sys
import json
import time
import urllib.request
import urllib.error
from sqlalchemy import create_engine, text
from database import mysql_engine as engine

# AI Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        config_path = os.path.join(os.path.dirname(__file__), "venv/backup_scripts/groq_config_auto.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                GROQ_API_KEY = config.get("api_key")
    except: pass

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

def call_groq(prompt, max_tokens=1000):
    """Full Groq API implementation with retry logic from your original scripts."""
    if not GROQ_API_KEY:
        return None
        
    data = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    
    json_data = json.dumps(data).encode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "User-Agent": "Mozilla/5.0"
    }
    
    req = urllib.request.Request(GROQ_API_URL, data=json_data, headers=headers)
    
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            if e.code == 429: # Rate limit
                wait_time = (attempt + 1) * 3
                print(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else: raise
        except Exception:
            if attempt < 2: time.sleep(2)
            else: raise
    return None

def call_ollama(prompt):
    """Full Ollama implementation for local AI generation."""
    try:
        import ollama
        response = ollama.chat(
            model='llama3.1:latest',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.7, 'num_predict': 500}
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return None

def enhance_question(question_id, question_text, options, answer, force_ollama=False):
    """
    Complete logic for generating Difficulty, Area, and Explanation.
    Uses your exact prompt format from import_mech_questions.py.
    """
    prompt = f"""Analyze this technical question and provide:

Question: {question_text}
Options: {options}
Correct Answer: {answer}

Provide in this EXACT format:

DIFFICULTY: [number from 1-10 where 1-3=basic concepts, 4-7=moderate application, 8-10=advanced analysis]
AREA: [Specific Technical Area]
EXPLANATION: [Detailed 300-500 word explanation showing why the answer is right and why others are wrong]

Return ONLY these three lines, nothing else."""

    response = None
    if not force_ollama:
        response = call_groq(prompt)
    
    if not response:
        print("Falling back to Ollama...")
        response = call_ollama(prompt)
        
    return response

def parse_ai_response(response):
    """Parses standard labels from AI output."""
    if not response: return None
    
    lines = response.strip().split('\n')
    data = {"difficulty_level": 5, "area": "General", "explanation": ""}
    
    for line in lines:
        if line.upper().startswith('DIFFICULTY:'):
            try:
                val = line.split(':')[1].strip().split()[0]
                data["difficulty_level"] = max(1, min(10, int(val)))
            except: pass
        elif line.upper().startswith('AREA:'):
            data["area"] = line.split(':', 1)[1].strip()
        elif line.upper().startswith('EXPLANATION:'):
            data["explanation"] = line.split(':', 1)[1].strip()
            
    # Map level to text
    lvl = data["difficulty_level"]
    data["difficulty_text"] = 'easy' if lvl <= 3 else 'medium' if lvl <= 7 else 'hard'
    
    return data

def process_batch(query_filter="explanation IS NULL OR explanation = ''", limit=10):
    """Logic for batch processing questions missing explanations."""
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT id, question, option_a, option_b, option_c, option_d, correct_answer FROM questions WHERE {query_filter} LIMIT {limit}"))
        rows = result.fetchall()
        
        for row in rows:
            print(f"Processing QID: {row[0]}")
            options = f"A: {row[2]}, B: {row[3]}, C: {row[4]}, D: {row[5]}"
            raw_ai = enhance_question(row[0], row[1], options, row[6])
            parsed = parse_ai_response(raw_ai)
            
            if parsed and parsed["explanation"]:
                conn.execute(text("""
                    UPDATE questions 
                    SET explanation = :exp, area = :area, difficulty = :diff, difficulty_level = :dl
                    WHERE id = :id
                """), {
                    "exp": parsed["explanation"], "area": parsed["area"], 
                    "diff": parsed["difficulty_text"], "dl": parsed["difficulty_level"], 
                    "id": row[0]
                })
                conn.commit()
                print(f"Success for QID: {row[0]}")
            else:
                print(f"Failed for QID: {row[0]}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        process_batch()
    else:
        print("AI Engine ready. Use --batch to start processing.")
