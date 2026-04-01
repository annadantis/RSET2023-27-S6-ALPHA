from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
import whisper 
import ollama  
import json    
import os
import random
import traceback
import sys
import smtplib
from email.message import EmailMessage
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, date, timedelta, timezone, time as dt_time
import pymysql
import time
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import re
from typing import List, Dict, Optional, Any
from question_generator import generate_questions_ai

# Timezone helper for IST (UTC+5:30)
def get_ist_now():
    return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)

def get_ist_date():
    return get_ist_now().date()

from database import get_db, init_db, test_connection, SessionLocal, mysql_engine as engine

# --- MODELS ---
class UserAuth(BaseModel):
    username: str # Changed to email or username depending on logic, keeping username for backward compat
    password: str
    branch: Optional[str] = None
    role: Optional[str] = 'student'

class SendOTPRequest(BaseModel):
    email: str
    username: str
    password: str
    branch: Optional[str] = None
    role: Optional[str] = 'student'

class VerifyOTPRequest(BaseModel):
    email: str
    otp_code: str

class ForgotPasswordRequest(BaseModel):
    email: str
    username: str
    branch: Optional[str] = None
    role: str = 'student'
    secret_code: Optional[str] = None

class ResetPasswordRequest(BaseModel):
    email: str
    otp_code: str
    new_password: str

class QuizRequest(BaseModel):
    username: str
    category: str
    target_branch: Optional[str] = None # Added for practice mode

class AnswerSubmission(BaseModel):
    username: str
    category: str
    question_id: int
    user_answer: str

class QuizCompleteSubmission(BaseModel):
    username: str
    category: str
    score: int
    total_questions: int
    target_branch: Optional[str] = None # Added for practice mode
    weak_area: Optional[str] = None # Added for analytics
    answers: Optional[List[Dict[str, Any]]] = None # Detailed breakdown

class QuizResult(BaseModel):
    username: str
    category: str
    score: int
    area: str

class UpdateBranchRequest(BaseModel):
    username: str
    branch: str

class GDBonusRequest(BaseModel):
    username: str
    gd_score: float  # numeric score (0–10) from the GD session

# Load Whisper model once at startup
stt_model = whisper.load_model("base")

def replenish_interview_questions(db: Session, branch: str = "COMMON", needed: int = 5):
    """
    Background worker to generate interview questions using the local model
    when the database pool runs low for a specific branch.
    """
    if needed <= 0: return

    print(f"🤖 [Background] Replenishing {needed} interview questions for {branch}...")
    prompt = f"""
    You are an expert HR and technical interviewer. Provide exactly {needed} difficult, high-quality interview questions for a fresh graduate in the '{branch}' engineering branch.
    Ensure questions are unique and challenging.
    
    Provide the output as a JSON array where each element is an object with:
    - "question": The interview question.
    - "ideal_answer": A concise (1-2 sentences) ideal answer to look out for.
    
    Return ONLY valid JSON format. Example:
    [
      {{"question": "How do you optimize a SQL query?", "ideal_answer": "Use indexes properly, avoid SELECT *, and analyze execution plans."}}
    ]
    """
    
    try:
        response = ollama.generate(model='llama3', prompt=prompt, format='json')
        questions_json = json.loads(response['response'])
        
        # Handle dict wrapping case `{"questions": [...]}`
        if isinstance(questions_json, dict) and len(questions_json.keys()) >= 1:
            key = list(questions_json.keys())[0]
            if isinstance(questions_json[key], list):
                questions_json = questions_json[key]
                
        if isinstance(questions_json, list):
            from models import InterviewQuestion
            added = 0
            for q_obj in questions_json:
                q_text = q_obj.get("question", "").strip()
                ans_text = q_obj.get("ideal_answer", "").strip()
                
                if q_text and ans_text:
                    new_q = InterviewQuestion(branch=branch.upper(), question=q_text, ideal_answer=ans_text)
                    db.add(new_q)
                    added += 1
            db.commit()
            print(f"✅ [Background] Successfully added {added} new interview questions for {branch}.")
        else:
            print(f"⚠️ [Background] Unexpected output format from AI for {branch} interview questions.")
    except Exception as e:
        print(f"❌ [Background] Failed to replenish interview questions for {branch}: {e}")
        db.rollback()

def check_week_gate(username: str, db: Session):
    """
    Check if the user is 'locked' from progressing due to missing requirements.
    Gate: Must finish previous days before today's quiz is available if they've missed more than 2 days?
    Or simply returns the status.
    """
    try:
        today = get_ist_date()
        monday = today - timedelta(days=today.weekday())
        
        # Count dual completions (Apt + Tech) this week
        dual_completion_res = db.execute(
            text("""
                SELECT COUNT(*) FROM (
                    SELECT DATE(timestamp) as d
                    FROM results 
                    WHERE username = :u 
                    AND category IN ('APTITUDE', 'TECHNICAL')
                    AND timestamp >= :monday
                    GROUP BY DATE(timestamp)
                    HAVING COUNT(DISTINCT category) = 2
                ) as t
            """),
            {"u": username, "monday": monday}
        ).fetchone()
        
        days_done = dual_completion_res[0] or 0
        
        # Check GD and Interview this week
        activity_counts = db.execute(
            text("""
                SELECT category, COUNT(*) as count
                FROM results 
                WHERE username = :u 
                AND timestamp >= :monday
                AND category IN ('GD', 'INTERVIEW')
                GROUP BY category
            """),
            {"u": username, "monday": monday}
        ).fetchall()
        
        stats = {row[0].upper(): row[1] for row in activity_counts}
        gd_done = stats.get('GD', 0) >= 1
        int_done = stats.get('INTERVIEW', 0) >= 1
        
        total_required_days = today.weekday() # How many days should have been done by now
        is_locked = (days_done < total_required_days - 1) # Example: allow 1 day grace
        
        return {
            "days_done": days_done,
            "gd_done": gd_done,
            "interview_done": int_done,
            "is_locked": is_locked,
            "required_quizzes": total_required_days
        }
    except Exception as e:
        print(f"Error in check_week_gate: {e}")
        return {"is_locked": False}

def process_weekly_level_up(username: str, db: Session):
    """
    Fluid Level-Up Logic
    - Rule: Must complete 7 days of daily quizzes + 1 GD + 1 Interview to unlock next level.
    - Trigger: Anytime the requirement is met since the last Level Update.
    """
    try:
        today = get_ist_date()
        
        # Get user
        user_result = db.execute(
            text("SELECT aptitude_level, technical_level, last_level_update FROM users WHERE username = :u"),
            {"u": username}
        )
        user = user_result.fetchone()
        if not user: return False
        
        apt_lvl, tech_lvl, last_update = user
        # Default last_update to 7 days ago if null to allow first-time check
        if not last_update:
            last_update = datetime.now() - timedelta(days=7)
            
        # 1. Aggregate Activity percentage since LAST UPDATE
        activity_counts = db.execute(
            text("""
                SELECT category, COUNT(*) as count, SUM(score) as total_s, SUM(total_questions) as total_q
                FROM results 
                WHERE username = :u 
                AND timestamp >= :last_up
                GROUP BY category
            """),
            {"u": username, "last_up": last_update}
        ).fetchall()
        
        # Calculate percentage: (total_score / total_questions) * 100
        stats = {}
        for row in activity_counts:
            cat = row[0].upper()
            cnt = row[1]
            t_s = row[2] or 0
            t_q = row[3] or 1 # prevent zero division
            
            # If total_questions is missing or 0 in DB, default to out of 10 for quizzes, 100 for GD/Interview
            if t_q == 0 or t_q == 1:
                 t_q = 100 if cat in ['GD', 'INTERVIEW'] else (cnt * 10)
                 
            percent = (t_s / t_q) * 100
            stats[cat] = {"count": cnt, "avg_percent": percent}

        # Count distinct days where both Aptitude and Technical were completed since last update
        dual_completion_res = db.execute(
            text("""
                SELECT COUNT(*) FROM (
                    SELECT DATE(timestamp) as d
                    FROM results 
                    WHERE username = :u 
                    AND category IN ('APTITUDE', 'TECHNICAL')
                    AND timestamp >= :last_up
                    GROUP BY DATE(timestamp)
                    HAVING COUNT(DISTINCT category) = 2
                ) as t
            """),
            {"u": username, "last_up": last_update}
        ).fetchone()
        
        days_completed = dual_completion_res[0] or 0

        # 2. Strict Requirements Check (Since last update)
        gd_sessions = stats.get('GD', {}).get('count', 0)
        interview_sessions = stats.get('INTERVIEW', {}).get('count', 0)

        # Percentages for quality check
        apt_percent = stats.get('APTITUDE', {}).get('avg_percent', 0)
        tech_percent = stats.get('TECHNICAL', {}).get('avg_percent', 0)
        gd_percent = stats.get('GD', {}).get('avg_percent', 0)
        interview_percent = stats.get('INTERVIEW', {}).get('avg_percent', 0)

        # Require 7 distinct days + 75% across required categories
        can_level_up_apt = (days_completed >= 7 and apt_percent >= 75)
        can_level_up_tech = (days_completed >= 7 and tech_percent >= 75 and gd_percent >= 75 and interview_percent >= 75)

        # 3. Level Up Logic
        new_apt_lvl = apt_lvl
        if can_level_up_apt and apt_lvl < 4:
            new_apt_lvl += 1
            
        new_tech_lvl = tech_lvl
        if can_level_up_tech and tech_lvl < 4:
            new_tech_lvl += 1
            new_tech_lvl += 1

        level_up_occurred = (new_apt_lvl > apt_lvl or new_tech_lvl > tech_lvl)

        # 4. Save to weekly_stats
        overall_avg = (apt_percent + tech_percent) / 2 if (apt_percent > 0 and tech_percent > 0) else (apt_percent or tech_percent)
        
        db.execute(
            text("""
                INSERT INTO weekly_stats (username, week_start_date, avg_score, is_level_up, total_activities)
                VALUES (:u, :d, :s, :lu, :cnt)
            """),
            {
                "u": username,
                "d": today - timedelta(days=6), # Record the starting Monday
                "s": overall_avg,
                "lu": 1 if level_up_occurred else 0,
                "cnt": days_completed + gd_sessions + interview_sessions
            }
        )

        # 5. Update User Levels
        db.execute(
            text("""
                UPDATE users SET 
                aptitude_level = :al, 
                technical_level = :tl, 
                last_level_update = NOW() 
                WHERE username = :u
            """),
            {"al": new_apt_lvl, "tl": new_tech_lvl, "u": username}
        )
        
        db.commit()
        return level_up_occurred

    except Exception as e:
        print(f"❌ Error in weekly level up: {e}")
        db.rollback()
        return False

def get_user_level(db: Session, username: str, category: str):

    """Get user's current difficulty level"""
    result = db.execute(
        text("SELECT aptitude_level, technical_level FROM users WHERE username = :username"),
        {"username": username}
    )
    user = result.fetchone()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    level = user[0] if category.lower() == "aptitude" else user[1]
    
    # Map level to difficulty
    if level == 1:
        return "Easy"
    elif level == 2:
        return "Medium"
    elif level == 3:
        return "Hard"
    else:
        return "Company-level"

def get_todays_questions(db: Session, username: str, category: str, target_branch: str = None, for_date: date = None):
    """Get 10 questions for a specific date (defaults to today) - avoiding repeats from previous days"""
    today = for_date or get_ist_date()
    
    # If target_branch is provided and DOES NOT MATCH user's branch, do NOT cache/use daily_quiz
    # This is "Practice Mode"
    
    user_result = db.execute(
        text("SELECT branch FROM users WHERE username = :username"),
        {"username": username}
    )
    user_row = user_result.fetchone()
    user_actual_branch = user_row[0] if user_row else None

    is_practice_mode = False
    if category.lower() == "technical" and target_branch and target_branch != user_actual_branch:
        is_practice_mode = True
        print(f"🎯 Practice Mode: User {username} ({user_actual_branch}) practicing {target_branch}")
    
    # Check if user already has questions for today (ONLY if not practice mode)
    if not is_practice_mode:
        result = db.execute(
            text("""
                SELECT question_ids FROM daily_quiz 
                WHERE username = :username 
                AND category = :category 
                AND quiz_date = :today
            """),
            {"username": username, "category": category.lower(), "today": today}
        )
        
        existing = result.fetchone()
        
        if existing and existing[0]:
            # Return existing question IDs, but VERIFY they still exist
            # This prevents returning fewer than 10 questions if some were deleted during updates
            try:
                id_list = [int(x) for x in existing[0].split(",")]
                placeholders = ",".join([str(x) for x in id_list])
                verify_result = db.execute(text(f"SELECT id FROM questions WHERE id IN ({placeholders})")).fetchall()
                verified_ids = [r[0] for r in verify_result]
                
                if len(verified_ids) == len(id_list) and len(verified_ids) >= 1:
                    return verified_ids
                else:
                    print(f"⚠️ Cache mismatch for {username}: {len(verified_ids)}/{len(id_list)} IDs found. Regenerating...")
            except Exception as e:
                print(f"⚠️ Error verifying cached IDs: {e}. Regenerating...")
    
    # Get all past questions to avoid repetition (Skip for practice mode to allow unlimited practice?)
    # Let's keep avoiding repetition for practice mode too to keep it fresh
    past_quizzes = db.execute(
        text("""
            SELECT question_ids FROM daily_quiz 
            WHERE username = :username 
            AND category = :category
        """),
        {"username": username, "category": category.lower()}
    ).fetchall()

    seen_ids = set()
    for row in past_quizzes:
        if row[0]:
            for qid in row[0].split(','):
                try:
                    seen_ids.add(int(qid))
                except ValueError:
                    pass

    # Get user level for difficulty
    difficulty = get_user_level(db, username, category)

    # Fetch ALL candidate questions for this category & difficulty
    if category.lower() == "technical":
        # Use target_branch if provided, otherwise user's actual branch
        branch_to_use = target_branch if target_branch else user_actual_branch
        
        # Map AEI to ECE for technical questions
        if branch_to_use and branch_to_use.upper() == 'AEI':
            branch_to_use = 'ECE'
            
        if branch_to_use:
            # Fetch candidate questions where the requested branch is in the comma-separated branch list
            # We use FIND_IN_SET and REPLACE to handle possible spaces in the DB values
            candidates_result = db.execute(
                text("""
                    SELECT id FROM questions 
                    WHERE category = :category 
                    AND LOWER(difficulty) = LOWER(:difficulty)
                    AND (
                        LOWER(branch) = LOWER(:branch)
                        OR FIND_IN_SET(LOWER(:branch), REPLACE(LOWER(branch), ' ', ''))
                    )
                """),
                {"category": category.lower(), "difficulty": difficulty, "branch": branch_to_use.upper()}
            )
        else:
            # No branch set - return empty (user must select branch)
            candidates_result = db.execute(
                text("""
                    SELECT id FROM questions 
                    WHERE category = :category 
                    AND LOWER(difficulty) = LOWER(:difficulty)
                    AND 1=0
                """),
                {"category": category.lower(), "difficulty": difficulty}
            )
    else:
        # Aptitude/GD - pull 'Common' branch, empty string, or NULL branch questions
        candidates_result = db.execute(
            text("""
                SELECT id FROM questions 
                WHERE LOWER(category) = LOWER(:category) 
                AND LOWER(difficulty) = LOWER(:difficulty)
                AND (LOWER(branch) = 'common' OR branch IS NULL OR branch = '')
            """),
            {"category": category, "difficulty": difficulty}
        )

    
    all_candidate_ids = [row[0] for row in candidates_result.fetchall()]
    
    # FALLBACK: If fewer than 10 questions found for specific difficulty, try other difficulties for this branch/category
    if len(all_candidate_ids) < 10:
        print(f"🔍 Only {len(all_candidate_ids)} questions for {category} at {difficulty} level. Pulling from other difficulties...")
        
        if category.lower() == "technical" and branch_to_use:
            # Pull ALL questions for this branch regardless of difficulty
            fallback_result = db.execute(
                text("""
                    SELECT id FROM questions 
                    WHERE category = :category 
                    AND (
                        LOWER(branch) = LOWER(:branch)
                        OR FIND_IN_SET(LOWER(:branch), REPLACE(LOWER(branch), ' ', ''))
                    )
                """),
                {"category": category.lower(), "branch": branch_to_use.upper()}
            )
            all_candidate_ids = [row[0] for row in fallback_result.fetchall()]
        elif category.lower() != "technical":
            # For Aptitude/GD, pull all 'Common', empty, or NULL questions regardless of difficulty
            fallback_result = db.execute(
                text("""
                    SELECT id FROM questions 
                    WHERE LOWER(category) = LOWER(:category) 
                    AND (LOWER(branch) = 'common' OR branch IS NULL OR branch = '')
                """),
                {"category": category}
            )
            all_candidate_ids = [row[0] for row in fallback_result.fetchall()]

            
        print(f"✅ Total available questions for {category}: {len(all_candidate_ids)}")

    
    # --- IMPROVED: WEEKLY TOPIC BALANCING LOGIC ---
    
    # 1. Identify Week Start (Monday)
    today_dt = get_ist_date()
    week_start = today_dt - timedelta(days=today_dt.weekday())
    week_start_ts = datetime.combine(week_start, dt_time.min)
    
    # 2. Get all distinct areas for this branch/category
    branch_val = branch_to_use.upper() if (category.lower() == "technical" and branch_to_use) else "COMMON"
    area_query = """
        SELECT DISTINCT area FROM questions 
        WHERE category = :category 
        AND (LOWER(branch) = LOWER(:branch) OR FIND_IN_SET(LOWER(:branch), REPLACE(LOWER(branch), ' ', '')) OR :branch = 'COMMON')
        AND area IS NOT NULL AND area != ''
    """
    distinct_raw = db.execute(text(area_query), {"category": category.lower(), "branch": branch_val}).fetchall()
    distinct_areas = [r[0] for r in distinct_raw]
    
    if not distinct_areas:
        distinct_areas = ["General"]
    
    # 3. Get current weekly distribution from results table
    distribution_query = """
        SELECT area, SUM(total_questions) as count 
        FROM results 
        WHERE username = :username 
        AND category = :category 
        AND timestamp >= :week_start
        GROUP BY area
    """
    db_counts = db.execute(text(distribution_query), {
        "username": username,
        "category": category.upper(),
        "week_start": week_start_ts
    }).fetchall()
    
    current_counts = {area: 0 for area in distinct_areas}
    for row in db_counts:
        area_name = row[0]
        if area_name in current_counts:
            current_counts[area_name] = int(row[1])
    
    # 4. Selection Algorithm: Deficit-Based (Highest Deficit First)
    # Target: 70 questions per week shared among topics
    target_per_area = 70.0 / len(distinct_areas)
    
    question_ids = []
    temp_counts = current_counts.copy()
    
    for _ in range(10):
        # Calculate deficits and pick highest
        deficits = []
        for area in distinct_areas:
            deficit = target_per_area - temp_counts[area]
            deficits.append((area, deficit))
        
        # Sort by deficit descending, add slight random shuffle to same-deficit areas
        random.shuffle(deficits)
        deficits.sort(key=lambda x: x[1], reverse=True)
        best_area = deficits[0][0]
        
        # Get candidate IDs for this SPECIFIC area
        area_candidates_query = """
            SELECT id FROM questions 
            WHERE category = :category 
            AND LOWER(area) = LOWER(:area)
            AND (LOWER(branch) = LOWER(:branch) OR FIND_IN_SET(LOWER(:branch), REPLACE(LOWER(branch), ' ', '')) OR :branch = 'COMMON')
        """
        area_candidates = [row[0] for row in db.execute(
            text(area_candidates_query),
            {"category": category.lower(), "area": best_area, "branch": branch_val}
        ).fetchall()]
        
        # Filter seen (avoiding repetition within the same daily quiz too)
        valid_candidates = [qid for qid in area_candidates if qid not in seen_ids and qid not in question_ids]
        
        selected_qid = None
        if valid_candidates:
            selected_qid = random.choice(valid_candidates)
        else:
            # Need more! Fallback: allow repeats if AI is disabled
            repeats = [qid for qid in area_candidates if qid not in question_ids]
            if repeats:
                selected_qid = random.choice(repeats)
        
        if selected_qid:
            question_ids.append(selected_qid)
            temp_counts[best_area] += 1
        else:
            # Absolute fallback: pick any available question from overall candidates
            remaining = [qid for qid in all_candidate_ids if qid not in question_ids]
            if remaining:
                pick = random.choice(remaining)
                question_ids.append(pick)

    # Truncate to 10 just in case
    question_ids = question_ids[:10]
    
    # Save for today (ONLY if NOT practice mode)
    if not is_practice_mode:
        db.execute(
            text("""
                INSERT INTO daily_quiz (username, category, quiz_date, question_ids)
                VALUES (:username, :category, :today, :ids)
                ON DUPLICATE KEY UPDATE question_ids = :ids
            """),
            {
                "username": username, 
                "category": category.lower(), 
                "today": today,
                "ids": ",".join(map(str, question_ids))
            }
        )
        db.commit()
    
    return question_ids

def get_questions_by_ids(db: Session, question_ids: list):
    """Fetch full question details"""
    placeholders = ",".join([str(id) for id in question_ids])
    
    result = db.execute(
        text(f"""
            SELECT id, question, option_a, option_b, option_c, option_d,
                   correct_answer, area, explanation, difficulty
            FROM questions 
            WHERE id IN ({placeholders})
        """)
    )
    
    rows = result.fetchall()
    
    # Map to dict for easy lookup
    questions_dict = {}
    for row in rows:
        ans = str(row[6]).strip().upper() if row[6] else "A"
        questions_dict[row[0]] = {
            "id": row[0],
            "question": row[1],
            "options": [row[2], row[3], row[4], row[5]],
            "answer": ans, # Frontend expects 'answer'
            "correct_answer": ans,
            "area": row[7] or "General",
            "explanation": row[8] or "No explanation provided.",
            "difficulty": row[9]
        }



    
    # Return in original order
    return [questions_dict[id] for id in question_ids if id in questions_dict]

def generate_question_explanation(db: Session, question_id: int):
    """Generate and save an AI explanation using Groq (via ai_engine) if it's missing."""
    from ai_engine import enhance_question, parse_ai_response
    
    result = db.execute(
        text("SELECT question, option_a, option_b, option_c, option_d, correct_answer, explanation FROM questions WHERE id = :id"),
        {"id": question_id}
    )
    row = result.fetchone()
    if not row:
        return
    
    question_text, opt_a, opt_b, opt_c, opt_d, correct_answer_letter, current_exp = row
    
    # If explanation is missing or placeholder, generate a new one
    is_placeholder = (
        not current_exp or 
        len(current_exp.strip()) < 50 or
        current_exp == "No explanation provided." or 
        current_exp == "Explanation generation failed."
    )
    
    if is_placeholder:
        print(f"🤖 [Background] Generating AI explanation for Q{question_id} using Groq...")
        options_text = f"A: {opt_a}, B: {opt_b}, C: {opt_c}, D: {opt_d}"
        
        try:
            # Use ai_engine to get high-quality explanation
            raw_ai = enhance_question(question_id, question_text, options_text, correct_answer_letter)
            parsed = parse_ai_response(raw_ai)
            
            if parsed and parsed.get("explanation"):
                new_explanation = parsed["explanation"]
                db.execute(
                    text("""
                        UPDATE questions 
                        SET explanation = :exp, area = :area, difficulty_level = :dl, difficulty = :dt 
                        WHERE id = :id
                    """),
                    {
                        "exp": new_explanation, 
                        "area": parsed["area"],
                        "dl": parsed["difficulty_level"],
                        "dt": parsed["difficulty_text"],
                        "id": question_id
                    }
                )
                db.commit()
                print(f"✅ [Background] Saved Groq explanation for Q{question_id}")
        except Exception as e:
            print(f"⚠️ [Background] Groq failed for Q{question_id}: {e}")

def ensure_explanations_exist(question_ids: list):
    """Ensure all questions in the list have AI explanations in parallel (Background Task)."""
    def job(qid):
        db = SessionLocal()
        try:
            generate_question_explanation(db, qid)
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(job, question_ids)

def validate_answer(db: Session, question_id: int, user_answer: str):
    """
    Validate user answer and return explanation.
    """
    result = db.execute(
        text("SELECT correct_answer, explanation FROM questions WHERE id = :id"),
        {"id": question_id}
    )
    
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Question not found")
    
    correct_answer_letter, current_exp = row
    correct_answer_letter = correct_answer_letter.strip().upper()
    
    # If it's missing or clearly too short/placeholder, generate on the fly
    is_placeholder = (
        not current_exp or 
        len(current_exp.strip()) < 50 or
        current_exp == "No explanation provided." or 
        current_exp == "Explanation generation failed."
    )
    
    if is_placeholder:
        print(f"🤖 [Auto-Refining] Triggering explanation generation for Q{question_id}")
        generate_question_explanation(db, question_id)
        # Fetch updated explanation
        result = db.execute(text("SELECT explanation FROM questions WHERE id = :id"), {"id": question_id})
        current_exp = result.fetchone()[0]

    user_answer = user_answer.strip().upper()
    is_correct = (user_answer == correct_answer_letter)
    
    return is_correct, correct_answer_letter, current_exp or "No explanation provided."

# --- FASTAPI SETUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*70)
    print("PLACEMENT APP - STARTING UP")
    print("="*70)
    
    try:
        with engine.connect() as conn:
            print("Database connection successful!")
        
        # Initialize DB tables
        init_db()
        print("Database tables initialized")

        # Initialize Whisper model
        print("Loading Whisper model...")
        global stt_model
        stt_model = whisper.load_model("base")
        print("Whisper model loaded!")
        
    except Exception as e:
        print(f"Startup Failed: {e}")
        traceback.print_exc()
        raise e
    
    print("\nFeatures enabled:")
    print("   - Adaptive Quiz System (10 questions/day)")
    print("   - Progressive Difficulty")
    print("   - Interview Practice")
    print("   - Performance Analytics")
    
    print("\nAPI Documentation:")
    print("   Swagger UI: http://localhost:8000/docs")
    print("\n" + "="*70 + "\n")
    
    yield
    
    print("\nShutting down gracefully...")

app = FastAPI(
    title="Placement Preparation Platform",
    description="AI-Powered Placement Prep with Adaptive Quiz System",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Import and include routers
from teacher_routes import router as teacher_router
from gd import router as gd_router
from news_routes import router as news_router

app.include_router(teacher_router)
app.include_router(gd_router, prefix="/gd_module", tags=["GD Module"])
app.include_router(news_router)

# --- AUTOMATION TRIGGER (For Manual Testing) ---
@app.post("/trigger_daily_generation", tags=["Automation"])
async def trigger_daily_generation():
    from automation_service import daily_question_job
    try:
        daily_question_job()
        return {"status": "success", "message": "Manual Question Generation Triggered."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gd_bonus", tags=["Quiz System"])
async def apply_gd_bonus(request: GDBonusRequest, db: Session = Depends(get_db)):
    """
    Apply the GD level bonus to technical_progress.

    RULES
    ─────
    • Only triggered when gd_score > 5.
    • +8% technical_progress logic.
    • Bonus is cumulative across sessions.
    """
    from models import User
    
    username  = request.username.strip()
    gd_score  = request.gd_score
    
    if gd_score <= 5:
        return {
            "bonus_applied": False,
            "level_up":      False,
            "message":       "Score <= 5 — no bonus awarded.",
        }

    # Apply the +8% bonus
    try:
        user_obj = db.query(User).filter(User.username == username).first()
        if not user_obj:
            raise HTTPException(status_code=404, detail="User not found")

        level_up = False
        user_obj.technical_progress = (user_obj.technical_progress or 0.0) + 8.0
        if user_obj.technical_progress >= 100.0:
            if user_obj.technical_level < 4:
                user_obj.technical_level    += 1
                user_obj.technical_progress  = 0.0
                level_up = True
            else:
                user_obj.technical_progress = 100.0   # cap at max level
        
        db.commit()
        print(f"✅ GD +8% bonus applied for {username}. Level up: {level_up}")

        return {
            "bonus_applied": True,
            "level_up":      level_up,
            "message": (
                "Level Up! You've advanced to the next difficulty."
                if level_up else
                "GD bonus applied! Progress +8%."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"GD BONUS ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- ROUTES ---

@app.post("/register", tags=["Authentication"])
async def register(user: UserAuth, db: Session = Depends(get_db)):
    """Register a new user"""
    from models import User
    try:
        existing = db.query(User).filter(User.username == user.username.strip()).first()
        if existing:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Explicitly assign branch and other fields
        new_user = User()
        new_user.username = user.username.strip()
        new_user.password_hash = user.password
        new_user.aptitude_level = 1
        new_user.technical_level = 1
        new_user.branch = user.branch
        new_user.role = user.role.lower() if user.role else 'student'
        
        db.add(new_user)
        db.commit()
        return {"status": "success", "message": "User registered successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/send-otp", tags=["Authentication"])
async def send_otp(request: SendOTPRequest, db: Session = Depends(get_db)):
    """Generate and send an OTP for registration"""
    from models import User, OTPVerification
    import re
    try:
        if request.role.lower() == 'student':
            email_pattern = r'^u\d{7}@rajagiri\.edu\.in$'
            if not re.match(email_pattern, request.email.lower().strip()):
                 raise HTTPException(status_code=400, detail="Student email must be in the format u*******@rajagiri.edu.in (u followed by 7 digits).")
            
        # Check if user already exists (by email or username)
        existing_user = db.query(User).filter(
            (User.username == request.username.strip()) | 
            (User.email == request.email.strip())
        ).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="User with this email or username already exists.")
            
        # Generate 6-digit OTP
        otp = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        
        # Save to database
        db.query(OTPVerification).filter(OTPVerification.email == request.email.strip()).delete()
        
        expires = datetime.now(timezone.utc) + timedelta(minutes=5)
        new_otp = OTPVerification(
            email=request.email.strip(),
            otp_code=otp,
            username=request.username.strip(),
            password_hash=request.password,
            branch=request.branch,
            role=request.role.lower(),
            expires_at=expires
        )
        db.add(new_otp)
        db.commit()
        
        # Send Email
        sender_email = os.environ.get("SMTP_EMAIL", "student1dev1rajagiri@gmail.com") # example default
        sender_password = os.environ.get("SMTP_PASSWORD", "")
        
        if not sender_password:
            print("WARNING: No SMTP_PASSWORD set, skipping email send and just printing OTP to console.")
            print(f"DEBUG ONLY: OTP for {request.email} is {otp}")
            return {"status": "success", "message": "OTP sent."}
            
        msg = EmailMessage()
        msg.set_content(f"Hello {request.username},\n\nYour AI Placement Assistant verification code is: {otp}\n\nThis code is valid for 5 minutes.\n\nThanks,\nPlacement Team")
        msg['Subject'] = 'Your Placement Assistant Verification Code'
        msg['From'] = sender_email
        msg['To'] = request.email.strip()
        
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"SMTP Error: {e}")
            raise HTTPException(status_code=500, detail="Failed to send email. Check SMTP credentials.")
        
        return {"status": "success", "message": "OTP sent to your email."}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/verify-otp-register", tags=["Authentication"])
async def verify_otp_register(request: VerifyOTPRequest, db: Session = Depends(get_db)):
    """Verify OTP and create user account"""
    from models import User, OTPVerification
    try:
        otp_record = db.query(OTPVerification).filter(OTPVerification.email == request.email.strip()).first()
        
        if not otp_record:
            raise HTTPException(status_code=400, detail="No pending registration found.")
            
        if otp_record.otp_code != request.otp_code.strip():
            raise HTTPException(status_code=400, detail="Invalid OTP code.")
            
        if datetime.now(timezone.utc) > otp_record.expires_at.replace(tzinfo=timezone.utc):
            db.delete(otp_record)
            db.commit()
            raise HTTPException(status_code=400, detail="OTP has expired. Request a new one.")
            
        # Create user
        new_user = User(
            username=otp_record.username,
            email=otp_record.email,
            password_hash=otp_record.password_hash,
            aptitude_level=1,
            technical_level=1,
            branch=otp_record.branch,
            role=otp_record.role
        )
        
        db.add(new_user)
        db.delete(otp_record)
        db.commit()
        return {"status": "success", "message": "Registration successful"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/forgot-password-otp", tags=["Authentication"])
async def forgot_password_otp(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Generate and send an OTP for password reset"""
    from models import User, OTPVerification
    import re
    try:
        if request.role.lower() == 'student':
            email_pattern = r'^u\d{7}@rajagiri\.edu\.in$'
            if not re.match(email_pattern, request.email.lower().strip()):
                 raise HTTPException(status_code=400, detail="Student email must be in the format u*******@rajagiri.edu.in (u followed by 7 digits).")

        # Verify strict matching Criteria
        query = db.query(User).filter(
            User.email == request.email.strip(),
            User.username == request.username.strip().upper(),
            User.role == request.role.lower()
        )
        
        if request.role.lower() == 'student':
            query = query.filter(User.branch == request.branch)
        elif request.role.lower() == 'teacher':
            # Teachers must provide the exact secret code to reset
            expected_secret = os.getenv("TEACHER_SECRET_CODE", "admin123")
            if request.secret_code != expected_secret:
                raise HTTPException(status_code=403, detail="Invalid Secret Access Code for Teacher.")
            
        user = query.first()
        
        if not user:
             raise HTTPException(status_code=404, detail="No matching account found with these exact details.")
             
        # Generate 6-digit OTP
        otp = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        
        # Save to database (we reuse the OTPVerification table, password_hash isn't strictly needed here but we fill it to pass constraints or use a dummy)
        db.query(OTPVerification).filter(OTPVerification.email == request.email.strip()).delete()
        
        expires = datetime.now(timezone.utc) + timedelta(minutes=5)
        new_otp = OTPVerification(
            email=request.email.strip(),
            otp_code=otp,
            username=user.username,
            password_hash="RESET", # Dummy value since we aren't creating a new user
            branch=user.branch,
            role=user.role,
            expires_at=expires
        )
        db.add(new_otp)
        db.commit()
        
        # Send Email
        sender_email = os.environ.get("SMTP_EMAIL", "student1dev1rajagiri@gmail.com") 
        sender_password = os.environ.get("SMTP_PASSWORD", "")
        
        if not sender_password:
            print("WARNING: No SMTP_PASSWORD set, skipping email send and printing OTP to console.")
            print(f"DEBUG ONLY: Password Reset OTP for {request.email} is {otp}")
            return {"status": "success", "message": "OTP sent."}
            
        msg = EmailMessage()
        msg.set_content(f"Hello {request.username.upper()},\n\nYou requested a password reset. Your verification code is: {otp}\n\nThis code is valid for 5 minutes.\n\nThanks,\nPlacement Team")
        msg['Subject'] = 'Your Password Reset Code'
        msg['From'] = sender_email
        msg['To'] = request.email.strip()
        
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"SMTP Error: {e}")
            raise HTTPException(status_code=500, detail="Failed to send email. Check SMTP credentials.")
        
        return {"status": "success", "message": "Password reset OTP sent to your email."}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/reset-password", tags=["Authentication"])
async def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Verify OTP and update user password"""
    from models import User, OTPVerification
    try:
        otp_record = db.query(OTPVerification).filter(OTPVerification.email == request.email.strip()).first()
        
        if not otp_record:
            raise HTTPException(status_code=400, detail="No pending password reset request found.")
            
        if otp_record.otp_code != request.otp_code.strip():
            raise HTTPException(status_code=400, detail="Invalid OTP code.")
            
        if datetime.now(timezone.utc) > otp_record.expires_at.replace(tzinfo=timezone.utc):
            db.delete(otp_record)
            db.commit()
            raise HTTPException(status_code=400, detail="OTP has expired. Request a new one.")
            
        # Find user and update password
        user = db.query(User).filter(User.email == request.email.strip()).first()
        if not user:
             raise HTTPException(status_code=404, detail="User account not found.")
             
        user.password_hash = request.new_password
        
        db.delete(otp_record)
        db.commit()
        return {"status": "success", "message": "Password successfully reset. You can now login."}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login", tags=["Authentication"])
async def login(user: UserAuth, db: Session = Depends(get_db)):
    """Login user via email or username"""
    try:
        # Check by email OR username (user.username actually holds the input)
        result = db.execute(
            text("SELECT username, password_hash FROM users WHERE (email = :username OR username = :username)"),
            {"username": user.username.strip()}
        )
        row = result.fetchone()
        
        if row and row[1] == user.password:
            # Trigger weekly level up check on login
            process_weekly_level_up(row[0], db) # use actual username
            return {"status": "success", "username": row[0]}
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_branch", tags=["User"])
async def update_branch(request: UpdateBranchRequest, db: Session = Depends(get_db)):
    """Update user's branch after registration"""
    from models import User
    try:
        user = db.query(User).filter(User.username == request.username.strip()).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        user.branch = request.branch
        
        # Clear technical quiz cache for this user so they get correct branch questions immediately
        print(f"🗑️ Clearing technical quiz cache for user: {request.username.strip()}")
        result = db.execute(
            text("DELETE FROM daily_quiz WHERE username = :username AND category = 'technical'"),
            {"username": request.username.strip()}
        )
        print(f"✅ Cache cleared. Rows affected: {result.rowcount}")
        
        db.commit()
        return {"status": "success", "message": f"Branch updated to {request.branch}"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_daily_quiz", tags=["Quiz System"])
async def get_daily_quiz(
    request: QuizRequest, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
    """
    Get questions for today (or the earliest missed day this week).

    MISSED-QUIZ LOGIC
    ─────────────────
    • Week  = Monday … Sunday (IST).
    • Scans from Monday up to (not including) today.
    • The FIRST day with no completed result for this category is returned
      as the catch-up target date.
    • After Sunday resets — prior-week misses are silently dropped.
    """
    try:
        today  = get_ist_date()
        monday = today - timedelta(days=today.weekday())  # Monday of current ISO week
        target_date = today  # default: no misses, serve today

        # ── Practice-mode detection ────────────────────────────────────
        user_row = db.execute(
            text("SELECT branch FROM users WHERE username = :u"),
            {"u": request.username.strip()}
        ).fetchone()
        user_actual_branch = user_row[0] if user_row else None

        is_practice_mode = (
            request.category.lower() == "technical" 
            and request.target_branch 
            and request.target_branch != user_actual_branch
        )

        # ── Missed-quiz scan (current week only, skip for practice) ────
        if not is_practice_mode:
            check_day = monday
            while check_day < today:
                completed = db.execute(
                    text("""
                        SELECT COUNT(*) FROM results 
                        WHERE username = :u 
                          AND category  = :cat
                          AND DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)) = :d
                    """),
                    {
                        "u":   request.username.strip(), 
                        "cat": request.category.upper(), 
                        "d":   check_day,
                    }
                ).scalar() or 0

                if completed == 0:
                    target_date = check_day
                    print(
                        f"📅 Catch-up quiz [{request.category}] for "
                        f"{request.username}: serving {target_date}"
                    )
                    break  # stop at the FIRST missed day
                
                check_day += timedelta(days=1)
        
        # ── Fetch / generate question IDs ──────────────────────────────
        question_ids = get_todays_questions(
            db, 
            request.username, 
            request.category, 
            request.target_branch, 
            for_date=target_date
        )
        
        background_tasks.add_task(ensure_explanations_exist, question_ids)
        questions = get_questions_by_ids(db, question_ids)
        
        is_catchup = target_date < today
        
        # ── Gate Check ─────────────────────────────────────────────────
        gate_status = check_week_gate(request.username, db)
        
        return {
            "status": "success",
            "date": str(target_date),
            "is_catchup": is_catchup,
            "catchup_for": str(target_date) if is_catchup else None,
            "difficulty": get_user_level(db, request.username, request.category),
            "questions": questions,
            "total": len(questions),
            "week_gate": gate_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check_answer", tags=["Quiz System"])
async def check_answer(submission: AnswerSubmission, db: Session = Depends(get_db)):
    """Check a single answer - returns immediate feedback"""
    try:
        is_correct, correct_answer, explanation = validate_answer(
            db, 
            submission.question_id, 
            submission.user_answer
        )
        
        return {
            "is_correct": is_correct,
            "correct_answer": correct_answer,
            "explanation": explanation,
            "user_answer": submission.user_answer
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit_quiz", tags=["Quiz System"])
async def submit_quiz(submission: QuizCompleteSubmission, db: Session = Depends(get_db)):
    """
    Submit completed quiz
    - Updates user level if score >= 70%
    """
    try:
        percentage = (submission.score / submission.total_questions) * 100
        
        # Determine if this is practice mode
        user_result = db.execute(
            text("SELECT branch FROM users WHERE username = :username"),
            {"username": submission.username}
        )
        user_row = user_result.fetchone()
        user_branch = user_row[0] if user_row else None
        
        is_practice_mode = False
        if submission.target_branch and submission.target_branch != user_branch and submission.category.upper() == "TECHNICAL":
            is_practice_mode = True
            print(f"🎯 Practice Mode Submission: Score not saved for {submission.username}")
        
        # Save result (ONLY if NOT practice mode)
        if not is_practice_mode:
            # Import Score and QuizAnswer locally to avoid circular dependencies if any
            from models import Score, QuizAnswer
            
            # Use provided weak area or default to "Daily Quiz"
            area_to_save = submission.weak_area if submission.weak_area else "Daily Quiz"
            
            # Use SQLAlchemy Model for consistent ID generation
            new_result = Score(
                username=submission.username,
                category=submission.category.upper(),
                score=submission.score,
                total_questions=submission.total_questions,
                area=area_to_save,
                timestamp=datetime.now(timezone.utc)
            )
            db.add(new_result)
            db.commit()
            db.refresh(new_result)

            # SAVE PER-QUESTION ANSWERS
            if submission.answers:
                from models import QuizAnswer
                for ans in submission.answers:
                    try:
                        # Use provided is_correct if integer, else recalculate
                        is_corr = ans.get('is_correct', 0)
                        
                        new_ans = QuizAnswer(
                            result_id=new_result.id,
                            question_id=ans.get('question_id') or ans.get('id'),
                            user_answer=str(ans.get('user_answer') or ans.get('user_selected') or ""),
                            is_correct=int(is_corr)
                        )
                        db.add(new_ans)
                    except Exception as e:
                        print(f"Error saving quiz answer: {e}")
            
            db.commit()
            
            # ── CUMULATIVE LEVELING LOGIC ──────────────────────────────────
            # +6% per quiz attempt where score >= 7 (Aptitude or Technical).
            # Only the category that was quizzed gets the progress boost.
            # Bonus is repeatable: every qualifying attempt adds +6%.
            from models import User
            level_up = False
            if submission.score >= 7:
                user_obj = db.query(User).filter(User.username == submission.username).first()
                if user_obj:
                    if submission.category.upper() == "APTITUDE":
                        user_obj.aptitude_progress = (user_obj.aptitude_progress or 0.0) + 6.0
                        if user_obj.aptitude_progress >= 100.0:
                            if user_obj.aptitude_level < 4:
                                user_obj.aptitude_level += 1
                                user_obj.aptitude_progress = 0.0   # reset for next level
                                level_up = True
                            else:
                                user_obj.aptitude_progress = 100.0  # cap at max level
                                
                    elif submission.category.upper() == "TECHNICAL":
                        user_obj.technical_progress = (user_obj.technical_progress or 0.0) + 6.0
                        if user_obj.technical_progress >= 100.0:
                            if user_obj.technical_level < 4:
                                user_obj.technical_level += 1
                                user_obj.technical_progress = 0.0
                                level_up = True
                            else:
                                user_obj.technical_progress = 100.0

                    db.commit()
            
            return {
                "status": "success",
                "score": submission.score,
                "total": submission.total_questions,
                "percentage": round(percentage, 1),
                "level_up": level_up,
                "message": (
                    "Level Up! You've advanced to the next difficulty." if level_up 
                    else "Progress +6%! Keep going!" if submission.score >= 7 
                    else "Keep practicing!"
                )
            }
        else:
            return {
                "status": "success",
                "score": submission.score,
                "total": submission.total_questions,
                "percentage": round(percentage, 1),
                "level_up": False,
                "message": f"Practice complete for {submission.target_branch}!"
            }
        
    except Exception as e:
        db.rollback()
        print(f"SUBMIT ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quiz_status/{username}/{category}", tags=["Quiz System"])
async def get_quiz_status(username: str, category: str, db: Session = Depends(get_db)):
    """Check if user has already taken today's quiz"""
    try:
        today = date.today()
        
        result = db.execute(
            text("""
                SELECT question_ids FROM daily_quiz 
                WHERE username = :username 
                AND category = :category 
                AND quiz_date = :today
            """),
            {"username": username, "category": category.lower(), "today": today}
        )
        
        existing = result.fetchone()
        has_quiz_today = existing is not None
        
        # Get user difficulty as fallback or current level
        difficulty = get_user_level(db, username, category)

        return {
            "has_quiz_today": has_quiz_today,
            "current_level": difficulty,
            "date": str(get_ist_date())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/weekly_report/{username}", tags=["Analytics"])
async def get_weekly_report(username: str, db: Session = Depends(get_db)):
    """Get performance analytics with daily and weekly aggregations"""
    try:
        # 1. Daily Aggregation for Aptitude and Technical combined (First attempt only)
        def get_daily_agg():
            res = db.execute(
                text("""
                    SELECT DATE(DATE_ADD(r.timestamp, INTERVAL '5:30' HOUR_MINUTE)) as day, AVG(r.score) as avg_score, r.category
                    FROM results r
                    INNER JOIN (
                        SELECT MIN(timestamp) as first_time, category
                        FROM results
                        WHERE username = :username AND category IN ('APTITUDE', 'TECHNICAL', 'GD', 'INTERVIEW')
                        AND DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE) >= DATE_SUB(DATE(DATE_ADD(NOW(), INTERVAL '5:30' HOUR_MINUTE)), INTERVAL 7 DAY)
                        GROUP BY DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)), category
                    ) sub ON r.timestamp = sub.first_time AND r.category = sub.category
                    WHERE r.username = :username AND r.category IN ('APTITUDE', 'TECHNICAL', 'GD', 'INTERVIEW')
                    GROUP BY day, r.category
                    ORDER BY day ASC
                """),
                {"username": username.strip()}
            )
            raw = res.fetchall()
            # Group by category for easier frontend use if needed, but the caller expects a single list of scores/times
            # Actually get_weekly_report caller seems to expect a single list for 'overall_daily'
            # Let's keep it as a combined list as it was, but including all categories
            return [{"score": round(float(r[1]), 1), "time": str(r[0]), "category": r[2]} for r in raw]

        # 2. Weekly Aggregation for GD and Interview (Last 4 Weeks)
        def get_weekly_agg(cat_name):
            res = db.execute(
                text("""
                    SELECT YEARWEEK(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)) as week, AVG(score) as avg_score, MIN(DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE))) as week_start
                    FROM results 
                    WHERE username = :username AND category = :category
                    AND DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE) >= DATE_SUB(DATE(DATE_ADD(NOW(), INTERVAL '5:30' HOUR_MINUTE)), INTERVAL 4 WEEK)
                    GROUP BY week
                    ORDER BY week ASC
                """),
                {"username": username.strip(), "category": cat_name}
            )
            return [{"score": round(float(r[1]), 1), "week_start": str(r[2])} for r in res.fetchall()]

        # 2b. Cumulative Weekly Growth (Sum of first attempts for all 4 categories)
        def get_cumulative_weekly():
            res = db.execute(
                text("""
                    SELECT YEARWEEK(DATE_ADD(r.timestamp, INTERVAL '5:30' HOUR_MINUTE)) as week, SUM(r.score) as total_score, MIN(DATE(DATE_ADD(r.timestamp, INTERVAL '5:30' HOUR_MINUTE))) as week_start
                    FROM results r
                    INNER JOIN (
                        SELECT MIN(timestamp) as first_time
                        FROM results
                        WHERE username = :username AND category IN ('APTITUDE', 'TECHNICAL', 'GD', 'INTERVIEW')
                        GROUP BY DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)), category
                    ) sub ON r.timestamp = sub.first_time
                    WHERE r.username = :username
                    GROUP BY week
                    ORDER BY week ASC
                """),
                {"username": username.strip()}
            )
            return [{"score": int(r[1]), "week_start": str(r[2])} for r in res.fetchall()]

        overall_daily = get_daily_agg()
        gd_weekly = get_weekly_agg("GD")
        interview_weekly = get_weekly_agg("INTERVIEW")
        cumulative_weekly = get_cumulative_weekly()

        # 3. Consistency Streak
        streak_query = db.execute(
            text("SELECT DISTINCT DATE(timestamp) as d FROM results WHERE username = :username ORDER BY d DESC"),
            {"username": username.strip()}
        ).fetchall()
        dates = [r[0] for r in streak_query]
        streak = 0
        curr = get_ist_date()
        if dates and curr not in dates and (curr - timedelta(days=1)) in dates:
            curr -= timedelta(days=1)
        while curr in dates:
            streak += 1
            curr -= timedelta(days=1)

        # 4. Strong Areas (Avg > 7.0)
        strong_res = db.execute(
            text("""
                SELECT area, AVG(score) as avg_s, COUNT(*) as count 
                FROM results 
                WHERE username = :username AND area IS NOT NULL AND area != '' AND area != 'Daily Quiz'
                GROUP BY area
            """),
            {"username": username.strip()}
        ).fetchall()
        
        strong_areas = [r[0] for r in strong_res if r[1] >= 7.0]
        
        # 5. Badges Logic
        badges = []
        # Topic Mastery (90% + min 5 attempts)
        for r in strong_res:
            if r[1] >= 9.0 and r[2] >= 5:
                badges.append({"name": f"{r[0]} Master", "icon": "emoji_events", "color": "gold"})
        
        # Consistency Badge
        if streak >= 7:
            badges.append({"name": "7-Day Streak", "icon": "whatshot", "color": "orange"})
            
        # GD Eloquent (Avg communication > 8.0 + min 3 sessions)
        gd_res = db.execute(
            text("SELECT AVG(communication_score), COUNT(*) FROM gd_evaluations WHERE username = :u"),
            {"u": username.strip()}
        ).fetchone()
        if gd_res and gd_res[0] and gd_res[0] >= 8.0 and gd_res[1] >= 3:
            badges.append({"name": "GD Eloquent", "icon": "record_voice_over", "color": "blue"})
            
        # Interview Ace (Avg overall > 8.5 + min 3 sessions)
        int_res = db.execute(
            text("SELECT AVG(score), COUNT(*) FROM results WHERE username = :u AND category = 'INTERVIEW'"),
            {"u": username.strip()}
        ).fetchone()
        if int_res and int_res[0] and int_res[0] >= 8.5 and int_res[1] >= 3:
            badges.append({"name": "Interview Ace", "icon": "work", "color": "purple"})

        # 7. Optimized Branch Rank Calculation
        branch_rank = 0
        user_res = db.execute(text("SELECT branch FROM users WHERE username = :u"), {"u": username.strip()}).fetchone()
        user_branch = user_res[0] if user_res else None
        if user_branch:
            # Use CTE for leaderboard to calculate rank based on average score
            rankings_res = db.execute(text("""
                SELECT username, rank_val FROM (
                    SELECT 
                        u.username,
                        AVG(r.score * 10) as avg_p,
                        RANK() OVER (ORDER BY AVG(r.score * 10) DESC) as rank_val
                    FROM users u
                    JOIN results r ON u.username = r.username
                    WHERE u.branch = :branch AND u.role = 'student'
                    GROUP BY u.username
                ) as leaderboard
                WHERE username = :username
            """), {"branch": user_branch, "username": username.strip()}).fetchone()
            
            if rankings_res:
                branch_rank = rankings_res[1]


        # 6. Calculate status and Readiness
        avg_res = db.execute(
            text("""
                SELECT AVG(score) FROM results 
                WHERE username = :username AND DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE) >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            """),
            {"username": username.strip()}
        )
        latest_avg = float(avg_res.scalar() or 0)
        
        # Readiness Score: (avg_score * 8) + (streak * 2) - capped at 100
        readiness_score = min(100, (latest_avg * 8.5) + (min(streak, 7) * 2.1))

        status = "Beginner"
        if latest_avg >= 8: status = "Expert"
        elif latest_avg >= 5: status = "Intermediate"

        return {
            "has_data": any([overall_daily, gd_weekly, interview_weekly]),
            "overall_daily": overall_daily,
            "gd_weekly": gd_weekly,
            "interview_weekly": interview_weekly,
            "cumulative_weekly": cumulative_weekly,
            "status": status,
            "current_performance": round(latest_avg, 1),
            "streak": streak,
            "strong_areas": strong_areas,
            "readiness_score": round(readiness_score, 1),
            "badges": badges,
            "streak": streak,
            "branch_rank": branch_rank,
            "week_gate": check_week_gate(username.strip(), db)
        }
    except Exception as e:
        print(f"Error in weekly_report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/daily_report/{username}", tags=["Analytics"])
async def get_daily_report(username: str, date_str: Optional[str] = None, db: Session = Depends(get_db)):
    """Get the daily report for a specific date (defaults to today)"""
    try:
        if date_str:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            target_date = get_ist_date()

        report: Dict[str, Any] = {
            "date": str(target_date),
            "TECHNICAL": [],
            "APTITUDE": [],
            "INTERVIEW": [],
            "GD": []
        }

        # 1. Fetch Technical, Aptitude, and Interview results from `results` table
        # We use a date range to handle potential timezone shifts (UTC/IST)
        results_query = db.execute(
            text("""
                SELECT id, category, score, total_questions, area, timestamp, confidence 
                FROM results 
                WHERE username = :u 
                AND (
                    DATE(timestamp) = :d OR 
                    DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)) = :d
                )
            """),
            {"u": username.strip(), "d": target_date}
        ).fetchall()

        for row in results_query:
            cat = row[1].upper()
            if cat in report:
                confidence_obj = {}
                if row[6]:
                    try:
                        confidence_obj = json.loads(row[6])
                    except:
                        pass
                
                # Special handling for GD to match frontend EXPECTATIONS in DailyReportScreen
                if cat == "GD":
                    gd_extra = db.execute(text("SELECT final_score, content_score, communication_score, topic_id FROM gd_results WHERE `result_id` = :rid"), {"rid": row[0]}).fetchone()
                    
                    report[cat].append({
                        "id": row[0],
                        "score": float(row[2]) if row[2] else 0.0,
                        "final_score": float(gd_extra[0]) if gd_extra else (float(row[2]) * 10), # fallback
                        "content_score": float(gd_extra[1]) if gd_extra else 0.0,
                        "communication_score": float(gd_extra[2]) if gd_extra else 0.0,
                        "topic_id": str(gd_extra[3]) if gd_extra else "",
                        "area": row[4] or "Group Discussion",
                        "timestamp": str(row[5])
                    })
                else:
                    report[cat].append({
                        "id": row[0],
                        "score": float(row[2]) if row[2] else 0.0,
                        "total_questions": int(row[3]) if row[3] else 0,
                        "area": row[4] or "",
                        "timestamp": str(row[5]),
                        "confidence_metrics": confidence_obj.get("metrics", {}) if isinstance(confidence_obj, dict) else {},
                        "feedback": confidence_obj.get("feedback", "") if isinstance(confidence_obj, dict) else ""
                    })

        # 2. GD fallback for OLD records (those not in results table with category 'GD')
        if not report["GD"]:
            try:
                gd_query = db.execute(
                    text("""
                        SELECT id, final_score, communication_score, content_score, topic_id, timestamp
                        FROM gd_results 
                        WHERE username = :u 
                        AND (
                            DATE(timestamp) = :d OR 
                            DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)) = :d
                        )
                    """),
                    {"u": username.strip(), "d": target_date}
                ).fetchall()
    
                for row in gd_query:
                    report["GD"].append({
                        "id": row[0], # NOTE: This refers to gd_results.id, which might cause issues in session_detail if not handled
                        "final_score": float(row[1]) if row[1] else 0.0,
                        "communication_score": float(row[2]) if row[2] else 0.0,
                        "content_score": float(row[3]) if row[3] else 0.0,
                        "topic_id": str(row[4]) if row[4] else "",
                        "timestamp": str(row[5]) if row[5] else "",
                    })
            except Exception as e:
                print(f"DEBUG: Could not fetch GD results, perhaps missing column. Error: {e}")

        return report

    except Exception as e:
        print(f"Error in daily_report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/leaderboard/{branch}", tags=["Leaderboard"])
async def get_branch_leaderboard(branch: str, db: Session = Depends(get_db)):
    """Fetch top 10 students in a branch based on Readiness Score"""
    try:
        # 1. Get all students in this branch
        users = db.execute(
            text("SELECT username, aptitude_level, technical_level FROM users WHERE branch = :b AND role = 'student'"),
            {"b": branch}
        ).fetchall()

        leaderboard = []
        today = get_ist_date()

        for user in users:
            uname = user[0]
            
            # 2. Calculate Streak (Past 7 days)
            streak = 0
            curr = today
            while True:
                count = db.execute(
                    text("SELECT COUNT(*) FROM results WHERE username = :u AND DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)) = :d"),
                    {"u": uname, "d": curr}
                ).fetchone()[0]
                if count > 0:
                    streak += 1
                    curr -= timedelta(days=1)
                else:
                    break
            
            # 3. Calculate Latest Avg
            avg_res = db.execute(
                text("""
                    SELECT AVG(score) FROM results 
                    WHERE username = :u AND timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                """),
                {"u": uname}
            ).scalar()
            latest_avg = float(avg_res or 0)

            # 3b. Calculate Penalty for Multiple Attempts (Retries)
            totals_res = db.execute(
                text("""
                    SELECT 
                        COUNT(*) as total_attempts,
                        COUNT(DISTINCT DATE(timestamp)) as unique_days
                    FROM results 
                    WHERE username = :u AND timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                """),
                {"u": uname}
            ).fetchone()
            
            unique_days = totals_res[1] or 0
            total_attempts = totals_res[0] or 0
            retries = max(0, total_attempts - unique_days)

            # 4. Final Readiness Score = Base - Retry Penalty
            readiness = min(100, (latest_avg * 8.5) + (min(streak, 7) * 2.1))
            readiness = max(0, readiness - (retries * 0.5))
            
            # 5. Get Badges Count
            badges_count = db.execute(
                text("SELECT COUNT(*) FROM results WHERE username = :u AND score >= 9"),
                {"u": uname}
            ).fetchone()[0] or 0

            leaderboard.append({
                "username": uname,
                "readiness_score": round(readiness, 1),
                "badges_count": badges_count,
                "level": user[2] # Technical level as proxy for expertise
            })

        # Sort by readiness score descending
        leaderboard.sort(key=lambda x: x['readiness_score'], reverse=True)
        
        return leaderboard[:10] # Return Top 10

    except Exception as e:
        print(f"LEADERBOARD ERROR: {e}")
        return []


@app.get("/dashboard/{username}", tags=["User"])
async def get_dashboard(username: str, db: Session = Depends(get_db)):
    """Get user dashboard data with task tracking and weekly progress"""
    from models import User, Score
    try:
        user = db.query(User).filter(User.username == username.strip()).first()
        if not user:
            raise HTTPException(status_code=404, detail="User find failed")

        # Trigger weekly checks
        process_weekly_level_up(username.strip(), db)
        
        # 1. Check Daily Completions (Scores recorded TODAY)
        today = get_ist_date()
        daily_res = db.execute(
            text("""
                SELECT category, SUM(score) as total_s, COUNT(*) as count 
                FROM results 
                WHERE username = :username AND DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)) = :today
                GROUP BY category
            """),
            {"username": username.strip(), "today": today}
        ).fetchall()
        
        daily_stats = {row[0].upper(): row[2] for row in daily_res}
        # Assuming each 'result' entry represents 1 session
        aptitude_done = daily_stats.get("APTITUDE", 0) >= 1
        tech_done = daily_stats.get("TECHNICAL", 0) >= 1
        interview_done = daily_stats.get("INTERVIEW", 0) >= 1
        gd_done = daily_stats.get("GD", 0) >= 1

        # 2. Check Weekly Completions (Strict CURRENT WEEK: Monday to Sunday)
        # Monday of this week: DATE_SUB(CURDATE(), INTERVAL WEEKDAY(CURDATE()) DAY)
        weekly_res = db.execute(
            text("""
                SELECT category, COUNT(*) as count 
                FROM results 
                WHERE username = :username 
                AND timestamp >= DATE_SUB(DATE(NOW()), INTERVAL WEEKDAY(DATE(NOW())) DAY)
                GROUP BY category
            """),
            {"username": username.strip()}
        ).fetchall()
        
        weekly_stats = {row[0].upper(): row[1] for row in weekly_res}
        gd_done = weekly_stats.get("GD", 0) >= 1
        interview_done = weekly_stats.get("INTERVIEW", 0) >= 1

        # 2. Count distinct days for each category this week
        tech_days = db.execute(text("""
            SELECT COUNT(DISTINCT DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE))) FROM results 
            WHERE username = :username AND category = 'TECHNICAL'
            AND DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE) >= DATE_SUB(DATE(DATE_ADD(NOW(), INTERVAL '5:30' HOUR_MINUTE)), INTERVAL WEEKDAY(DATE(DATE_ADD(NOW(), INTERVAL '5:30' HOUR_MINUTE))) DAY)
        """), {"username": username.strip()}).fetchone()[0] or 0
        
        apt_days = db.execute(text("""
            SELECT COUNT(DISTINCT DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE))) FROM results 
            WHERE username = :username AND category = 'APTITUDE'
            AND DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE) >= DATE_SUB(DATE(DATE_ADD(NOW(), INTERVAL '5:30' HOUR_MINUTE)), INTERVAL WEEKDAY(DATE(DATE_ADD(NOW(), INTERVAL '5:30' HOUR_MINUTE))) DAY)
        """), {"username": username.strip()}).fetchone()[0] or 0

        # 3. Calculate Weekly Progress (%)
        # Logic: 7 units for Tech + 7 units for Aptitude + 1 unit for GD + 1 unit for Interview = 16 units total
        progress_units = min(tech_days, 7) + min(apt_days, 7) + (1 if gd_done else 0) + (1 if interview_done else 0)
        weekly_progress = (progress_units / 16.0) * 100

        # 4. Weak Areas logic
        def get_top_weak_areas(cat_name):
            # 1. Total Distinct Days Check
            distinct_days_res = db.execute(
                text("SELECT COUNT(DISTINCT DATE(timestamp)) FROM results WHERE username = :username"),
                {"username": username.strip()}
            ).fetchone()
            
            total_days = distinct_days_res[0] or 0
            
            # 2. Strict Rolling 7-Day Window
            rolling_window_start = "DATE_SUB(NOW(), INTERVAL 7 DAY)"
            
            # Identify weak areas based ONLY on the last 7 days
            res = db.execute(
                text(f"""
                    SELECT 
                        area, 
                        SUM(score) as correct, 
                        SUM(total_questions) as total,
                        (SUM(score) / SUM(total_questions) * 100) as percentage
                    FROM results 
                    WHERE username = :username AND category = :category 
                    AND area != 'Daily Quiz' AND area IS NOT NULL
                    AND timestamp >= {rolling_window_start}
                    GROUP BY area 
                    HAVING percentage <= 70.0
                    ORDER BY percentage ASC 
                    LIMIT 3
                """),
                {"username": username.strip(), "category": cat_name}
            ).fetchall()
            
            return {
                "areas": [
                    {
                        "area": r[0], 
                        "correct": int(r[1]),
                        "total": int(r[2]),
                        "wrong": int(r[2]) - int(r[1]),
                        "percentage": round(float(r[3]), 1)
                    } for r in res
                ],
                "total_days": total_days,
                "status": "active" if total_days >= 3 else "collecting"
            }

        weak_areas_tech_data = get_top_weak_areas("TECHNICAL")
        weak_areas_apt_data = get_top_weak_areas("APTITUDE")

        # 5. Accuracy & Total Attempts (Include both results and gd_results)
        all_time_res = db.execute(
            text("""
                SELECT SUM(cnt), SUM(total_score)
                FROM (
                    SELECT COUNT(*) as cnt, SUM(score) as total_score FROM results WHERE username = :username
                    UNION ALL
                    SELECT COUNT(*) as cnt, SUM(overall_score) as total_score FROM gd_results WHERE username = :username AND `result_id` IS NULL
                ) combined
            """),
            {"username": username.strip()}
        ).fetchone()
        
        total_attempts = int(all_time_res[0] or 0)
        sum_scores = float(all_time_res[1] or 0)
        avg_score = sum_scores / total_attempts if total_attempts > 0 else 0
        accuracy = (avg_score / 10.0) * 100

        # 6. Recent Daily Performance (Strictly Current Week: Monday to Sunday)
        current_week_daily = {
            "aptitude": [],
            "technical": [],
            "interview": [],
            "gd": []
        }
        
        # Calculate most recent Monday
        today = get_ist_date()
        monday = today - timedelta(days=today.weekday())
        
        for cat in ["APTITUDE", "TECHNICAL", "INTERVIEW", "GD"]:
            res = db.execute(
                text("""
                    SELECT DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)) as day, MAX(score) as avg_score, DAYNAME(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)) as day_name
                    FROM results
                    WHERE username = :username AND category = :cat
                    AND DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)) >= :monday
                    GROUP BY day, day_name
                    ORDER BY day ASC
                """),
                {"username": username.strip(), "cat": cat, "monday": monday}
            ).fetchall()
            
            current_week_daily[cat.lower()] = [
                {"day": str(row[0]), "score": float(row[1]), "day_name": (row[2] or "Day")[:3]} 
                for row in res
            ]

        # 7. Last Week Daily Performance (Previous Monday to Sunday)
        last_week_daily = {
            "aptitude": [],
            "technical": [],
            "interview": [],
            "gd": []
        }
        last_monday = monday - timedelta(days=7)
        last_sunday = monday - timedelta(days=1)
        
        for cat in ["APTITUDE", "TECHNICAL", "INTERVIEW", "GD"]:
            res_lw = db.execute(
                text("""
                    SELECT DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)) as day, MAX(score) as avg_score, DAYNAME(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)) as day_name
                    FROM results
                    WHERE username = :username AND category = :cat
                    AND DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)) >= :last_monday 
                    AND DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)) <= :last_sunday
                    GROUP BY day, day_name
                    ORDER BY day ASC
                """),
                {"username": username.strip(), "cat": cat, "last_monday": last_monday, "last_sunday": last_sunday}
            ).fetchall()
            
            last_week_daily[cat.lower()] = [
                {"day": str(row[0]), "score": float(row[1]), "day_name": (row[2] or "Day")[:3]} 
                for row in res_lw
            ]

        # 8. Branch Rank Calculation
        if user.branch:
            # Optimized Rank Calculation inclusive of all students in the branch
            rankings_res = db.execute(text("""
                SELECT username, rank_val FROM (
                    SELECT 
                        u.username,
                        COALESCE(AVG(r.score * 10), 0) as avg_p,
                        RANK() OVER (ORDER BY COALESCE(AVG(r.score * 10), 0) DESC, u.created_at ASC) as rank_val
                    FROM users u
                    LEFT JOIN results r ON u.username = r.username
                    WHERE u.branch = :branch AND u.role = 'student'
                    GROUP BY u.username
                ) as leaderboard
                WHERE username = :username
            """), {"branch": user.branch, "username": username.strip()}).fetchone()
            
            if rankings_res:
                branch_rank = rankings_res[1]
            
            total_branch_students = db.execute(
                text("SELECT COUNT(*) FROM users WHERE branch = :branch AND role = 'student'"),
                {"branch": user.branch}
            ).scalar() or 0


        return {
            "id": user.id,
            "username": user.username,
            "aptitude_level": user.aptitude_level,
            "technical_level": user.technical_level,
            "branch": user.branch,
            "weak_areas_tech": weak_areas_tech_data["areas"],
            "weak_areas_apt": weak_areas_apt_data["areas"],
            "weak_areas_tech_status": weak_areas_tech_data["status"],
            "weak_areas_apt_status": weak_areas_apt_data["status"],
            "total_days": weak_areas_tech_data["total_days"],
            "tasks": {
                "aptitude_done": aptitude_done,
                "tech_done": tech_done,
                "gd_done": gd_done,
                "interview_done": interview_done
            },
            "weekly_progress": round(weekly_progress, 1),
            "total_attempts": total_attempts,
            "accuracy": round(accuracy, 1),
            "current_week_daily": current_week_daily,
            "last_week_daily": last_week_daily,
            "branch_rank": branch_rank,
            "total_branch_students": total_branch_students
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_confidence(transcript: str, duration: float):
    """
    Calculate confidence score based on speech metrics.
    Metrics: WPM (Speed), Filler Words, Answer Length, Tone (Sentiment)
    """
    words = transcript.split()
    word_count = len(words)
    
    if word_count == 0:
        return {
            "score": 0,
            "wpm": 0,
            "filler_count": 0,
            "sentiment_score": 0,
            "duration": round(duration, 1),
            "word_count": 0,
            "analysis": {
                "pace": "None",
                "length": "None",
                "fillers": "None"
            }
        }
    
    # 1. Speech Speed (WPM)
    if duration > 0:
        wpm = (word_count / duration) * 60
    else:
        wpm = 0
        
    if 110 <= wpm <= 160: 
        speed_score = 100
    elif (90 <= wpm < 110) or (160 < wpm <= 180): 
        speed_score = 70
    else: 
        speed_score = 40
    
    # 2. Filler Words detection
    filler_words = ["um", "uh", "like", "actually", "basically", "you know"]
    filler_count = sum(transcript.lower().count(f) for f in filler_words)
    filler_score = max(0, 100 - filler_count * 5)
    
    # 3. Answer Length
    if word_count > 60: 
        length_score = 100
    elif 30 <= word_count <= 60: 
        length_score = 80
    elif 10 <= word_count < 30: 
        length_score = 60
    else: 
        length_score = 40
    
    # 4. Keyword-based Sentiment (Tone)
    pos_words = ["believe", "sure", "definitely", "achieved", "implemented", "passion", "expert", "experience", "successfully", "impact", "growth", "challenge", "solved"]
    neg_words = ["maybe", "guess", "not sure", "possibly", "difficult", "struggled", "don't know", "hard"]
    sentiment_score = 60 # Confident baseline
    for w in words:
        clean_w = re.sub(r'[^\w]', '', w.lower())
        if clean_w in pos_words: sentiment_score += 5
        if clean_w in neg_words: sentiment_score -= 5
    sentiment_score = max(0, min(100, sentiment_score))
    
    # Final Weighted Average
    confidence = (speed_score + filler_score + length_score + sentiment_score) / 4
    
    return {
        "score": round(confidence),
        "wpm": round(wpm),
        "filler_count": filler_count,
        "sentiment_score": sentiment_score,
        "duration": round(duration, 1),
        "word_count": word_count,
        "analysis": {
            "pace": "Good" if speed_score == 100 else "Average" if speed_score == 70 else "Needs Improvement",
            "length": "Adequate" if length_score >= 80 else "Short",
            "fillers": "Low" if filler_score >= 90 else "Noticeable" if filler_score >= 70 else "High"
        }
    }

@app.post("/evaluate_interview", tags=["Interview Practice"])
async def evaluate(audio: UploadFile = File(...), username: str = Form("Anonymous"), db: Session = Depends(get_db)):
    """Evaluate interview audio response"""
    username = username.strip()
    temp_filename = f"temp_{audio.filename}"
    try:
        with open(temp_filename, "wb") as f:
            f.write(await audio.read())
        
        transcription_result = stt_model.transcribe(temp_filename)
        transcription = transcription_result.get("text", "").strip()
        segments = transcription_result.get("segments", [])
        duration = segments[-1].get("end", 0) if segments else 0

        # Calculate Speech Confidence
        confidence_data = calculate_confidence(transcription, duration)

        prompt = f"""
You are an expert Technical Interviewer. 

Question: "Explain final vs const in Dart."
Candidate's Answer: "{transcription}"

Instruction:
1. Evaluate the candidate's technical accuracy.
2. Provide feedback in ENGLISH only.
3. Return exactly ONE JSON object.

Return Format:
{{
  "score": "X/10",
  "feedback": "Concise English feedback here",
  "ideal_answer": "Model English explanation here"
}}

Rules:
- Return ONLY the JSON object.
- Language must be English.
"""
        
        response = ollama.generate(model='llama3.2', prompt=prompt, format='json') 
        try:
            data = json.loads(response['response'])
        except Exception:
            # Fallback to the robust extractor
            from ollama_eval import extract_json as safe_extract
            data = safe_extract(response['response'])
        
        # Extract numeric score (e.g., "7/10" -> 7)
        score_val = 5 # default
        if 'score' in data:
            match = re.search(r'(\d+(\.\d+)?)', str(data['score']))
            if match:
                score_val = float(match.group(1))

        # Persist the score to the results table using the Score model for reliable ID capture
        from models import Score, InterviewDetail
        
        new_score = Score(
            username=username,
            score=score_val,
            total_questions=10,
            category="INTERVIEW",
            area="Interview Practice", 
            confidence=json.dumps({
                "feedback": data.get("feedback", ""),
                "metrics": confidence_data
            }), 
            timestamp=get_ist_now()
        )
        db.add(new_score)
        db.commit()
        db.refresh(new_score)
        
        # Save technical breakdown
        detail = InterviewDetail(
            result_id=new_score.id,
            question="Explain final vs const in Dart.",
            user_answer=transcription,
            accuracy=data.get("score", "5/10"),
            ideal_answer=data.get("ideal_answer", ""),
            improvement="Focus on the differences between compile-time and run-time constants." 
        )
        db.add(detail)
        db.commit()

        # ── INTERVIEW LEVEL BONUS: +8% for score > 5 ──
        from models import User
        level_up = False
        if score_val > 5:
            user_obj = db.query(User).filter(User.username == username).first()
            if user_obj:
                user_obj.technical_progress = (user_obj.technical_progress or 0.0) + 8.0
                if user_obj.technical_progress >= 100.0:
                    if user_obj.technical_level < 4:
                        user_obj.technical_level += 1
                        user_obj.technical_progress = 0.0
                        level_up = True
                    else:
                        user_obj.technical_progress = 100.0  # cap at max level
                db.commit()

        return {
            "status": "success",
            "score": data.get("score", "5/10"),
            "feedback": data.get("feedback", ""),
            "confidence": confidence_data,
            "ideal_answer": data.get("ideal_answer", ""),
            "level_up": level_up
        }

    except Exception as e:
        if db: db.rollback() # Rollback in case of error during DB operation
        print(f"INTERVIEW ERROR: {e}")
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

import cv2
import numpy as np
from scipy.spatial import distance as dist

# --- GLOBAL SESSION STORAGE ---
session_answers = {}
session_metrics = {} # { username: { 'frames': 0, 'face': 0, 'gaze': 0, 'smile': 0, 'light': 0, 'multi': 0, 'box': 0 } }

# --- MEDIAPIPE INITIALIZATION ---
try:
    import mediapipe.solutions.face_mesh as mp_face_mesh
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe initialized successfully")
except Exception as e:
    try:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        MEDIAPIPE_AVAILABLE = True
        print("✅ MediaPipe loaded via python.solutions")
    except Exception:
        MEDIAPIPE_AVAILABLE = False
        mp_face_mesh = None
        print(f"⚠️  MediaPipe Solutions load error (Legacy session analyzer disabled): {e}")

# Indices for Behavioral Tracking
L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [133, 158, 160, 33, 144, 153]
IRIS_CENTER = 468 

# --- BEHAVIORAL HELPERS ---
def get_ear(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def analyze_video_session(video_path):
    if not MEDIAPIPE_AVAILABLE:
        return {"eye_contact": "N/A", "blinks_per_min": 0, "demeanor": "Unknown"}
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"eye_contact": "0%", "blinks_per_min": 0, "demeanor": "N/A"}

    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    total_frames, blink_count, contact_frames, consec_frames = 0, 0, 0, 0
    smile_frames = 0 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        total_frames += 1
        
        # Optimize: Analyze every 10th frame (approx 3 FPS) for significantly faster feedback
        if total_frames % 10 != 0: continue 
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        
        if res.multi_face_landmarks:
            pts = res.multi_face_landmarks[0].landmark
            
            # 1. EAR for Blinks
            le = np.array([[pts[i].x, pts[i].y] for i in L_EYE])
            re = np.array([[pts[i].x, pts[i].y] for i in R_EYE])
            ear = (get_ear(le) + get_ear(re)) / 2.0
            if ear < 0.18: # Optimized threshold
                consec_frames += 1
            else:
                if consec_frames >= 2: blink_count += 1
                consec_frames = 0
            
            # 2. Gaze Tracking (0.35 - 0.65 for natural eye movement)
            iris = pts[IRIS_CENTER]
            ratio = (iris.x - pts[33].x) / (pts[133].x - pts[33].x)
            if 0.35 < ratio < 0.65: contact_frames += 1

            # 3. Basic Demeanor Tracking
            m_left, m_right = pts[61], pts[291]
            mouth_width = dist.euclidean([m_left.x, m_left.y], [m_right.x, m_right.y])
            if mouth_width > 0.07: smile_frames += 1
            
    cap.release()
    duration_min = (total_frames / 30) / 60
    
    demeanor = "Confident/Positive" if smile_frames > (total_frames * 0.1) else "Serious/Focused"
    if contact_frames < (total_frames * 0.3): demeanor = "Anxious/Distracted"

    return {
        "eye_contact": f"{round((contact_frames/max(1,total_frames))*100, 1)}%",
        "blinks_per_min": round(blink_count / max(0.1, duration_min), 1),
        "demeanor": demeanor
    }

# --- UPDATED: NON-MCQ QUESTION LOADER ---
import csv
import random

def load_questions_by_difficulty(filename, level):
    questions = []
    if not os.path.exists(filename): 
        return []
    try:
        with open(filename, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                # Use capitalized keys to match CSV headers
                q_text = row.get("Question", "").strip()
                if not q_text:
                    continue
                
                # Loose MCQ check: only skip if it explicitly mentions "following options" in text
                if "which of the following" in q_text.lower():
                    continue
                
                try: 
                    q_diff = int(row.get("Difficulty", 1))
                except: 
                    q_diff = 1
                    
                if q_diff == level:
                    questions.append({
                        "question": q_text,
                        "difficulty": q_diff,
                        "area": row.get("Area", "Technical"),
                        "ideal_answer": row.get("Answer", ""), 
                        "explanation": "Standard logic applies."
                    })
        return questions
    except Exception as e: 
        print(f"Error loading questions: {e}")
        return []

@app.get("/get_questions/{username}/{category}")
async def get_questions(username: str, category: str, background_tasks: BackgroundTasks):
    db = SessionLocal()
    user = db.execute(
        text("SELECT * FROM users WHERE username=:u"), {"u": username.strip()}
    ).fetchone()
    
    cat = category.upper()
    level = getattr(user, "aptitude_level", 1) if cat == "APTITUDE" else getattr(user, "technical_level", 1) if user else 1
    branch = getattr(user, "branch", "COMMON") or "COMMON"
    branch = branch.upper()

    if cat == "INTERVIEW":
        from models import InterviewQuestion, UserAskedQuestion
        from sqlalchemy import func
        
        username_clean = username.strip()
        # Get IDs of questions already asked to this user
        asked_q_ids = [row[0] for row in db.query(UserAskedQuestion.question_id).filter(UserAskedQuestion.username == username_clean).all()]
        
        # Fetch 5 random unasked questions from DB for this branch
        query = db.query(InterviewQuestion).filter(InterviewQuestion.branch == branch)
        if asked_q_ids:
            query = query.filter(InterviewQuestion.id.notin_(asked_q_ids))
            
        questions = query.order_by(func.rand()).limit(5).all()
        
        formatted_qs = []
        if questions:
            for q in questions:
                formatted_qs.append({
                    "question": q.question,
                    "ideal_answer": q.ideal_answer,
                    "difficulty": 1,
                    "area": "Interview"
                })
                # Track that this user has seen this question
                db.add(UserAskedQuestion(username=username_clean, question_id=q.id))
            db.commit()
            
            # Replenish if the branch itself has very few total questions
            total_branch_qs = db.query(InterviewQuestion).filter(InterviewQuestion.branch == branch).count()
            if total_branch_qs < 20:
                background_tasks.add_task(replenish_interview_questions, db, branch, 10)
                
            # If we couldn't find 5 questions for *this user*, they are running out of fresh questions
            if len(questions) < 5:
                 background_tasks.add_task(replenish_interview_questions, db, branch, 5)

        else:
            # Fallback if DB has NO unasked questions for this branch
            background_tasks.add_task(replenish_interview_questions, db, branch, 10)
            
            # Try to grab some general COMMON questions they haven't seen
            fallback_query = db.query(InterviewQuestion).filter(InterviewQuestion.branch == "COMMON")
            if asked_q_ids:
                 fallback_query = fallback_query.filter(InterviewQuestion.id.notin_(asked_q_ids))
                 
            fallback = fallback_query.order_by(func.rand()).limit(5).all()
            for q in fallback:
                 formatted_qs.append({
                    "question": q.question,
                    "ideal_answer": q.ideal_answer,
                    "difficulty": 1,
                    "area": "Interview"
                 })
                 db.add(UserAskedQuestion(username=username_clean, question_id=q.id))
            db.commit()
            
            if len(formatted_qs) < 5:
                 # Absolute fallback if nothing works or DB is very empty
                 placeholders = [
                     {"question": f"Please introduce yourself and explain your interest in {branch}.", "ideal_answer": "Provide a solid introduction.", "difficulty": 1, "area": "HR"},
                     {"question": "Where do you see yourself in the next 5 years?", "ideal_answer": "Discuss career growth and stability.", "difficulty": 1, "area": "HR"},
                     {"question": "What are your greatest technical strengths?", "ideal_answer": "Be specific about technologies you know well.", "difficulty": 1, "area": "Technical"},
                     {"question": "Can you describe a challenging project you worked on?", "ideal_answer": "Use the STAR method: Situation, Task, Action, Result.", "difficulty": 1, "area": "Experience"},
                     {"question": "What interests you most about this branch/industry?", "ideal_answer": "Show passion and research about the field.", "difficulty": 1, "area": "Motivation"}
                 ]
                 # Fill up to 5 questions
                 while len(formatted_qs) < 5:
                     formatted_qs.append(placeholders[len(formatted_qs)])

        db.close()
        return {"questions": formatted_qs, "level": level}

    
    # Constants for csv
    APTITUDE_CSV = "datasets/enhanced_clean_general_aptitude_dataset.csv"
    TECHNICAL_CSV = "datasets/enhanced_cse_dataset.csv"
    
    file = APTITUDE_CSV if cat == "APTITUDE" else TECHNICAL_CSV
    qs = load_questions_by_difficulty(file, level)
    if len(qs) < 5: 
        qs = load_questions_by_difficulty(file, 1) + load_questions_by_difficulty(file, 2)
    random.shuffle(qs)
    db.close()
    return {"questions": [{"id": q.id, "question": q.question} for q in qs[:5]], "level": level}

# --- UPDATED: INTERVIEW EVALUATION ---

@app.post("/process_frame")
async def process_frame(username: str = Form(...), frame: UploadFile = File(...)):
    """
    Analyze a single camera frame for live interview warnings using OpenCV Haar Cascades.
    No MediaPipe dependency needed.
    """
    username = username.strip()
    temp_path = f"frame_{username}.jpg"
    content = await frame.read()
    with open(temp_path, "wb") as f:
        f.write(content)
        
    try:
        img = cv2.imread(temp_path)
        if img is None:
            print("DEBUG: Failed to read image")
            return {"warning": "FACE_NOT_DETECTED"}
            
        if username not in session_metrics:
            session_metrics[username] = {"frames": 0, "face": 0, "gaze": 0, "smile": 0, "light": 0, "multi": 0, "box": 0}
        
        session_metrics[username]["frames"] += 1

        # 1. Low Light Detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        if brightness < 40:
            session_metrics[username]["light"] += 1
            print(f"DEBUG: Low Light detected ({brightness:.1f})")
            return {"warning": "LOW_LIGHT"}

        # 2. Face Detection using OpenCV Haar Cascades (no MediaPipe needed)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        if len(faces) == 0:
            print("DEBUG: No face detected")
            return {"warning": "FACE_NOT_DETECTED"}
        
        session_metrics[username]["face"] += 1
        
        # 3. Multiple Faces Detection
        if len(faces) > 1:
            session_metrics[username]["multi"] += 1
            print("DEBUG: Multiple faces detected")
            return {"warning": "MULTIPLE_FACES"}
        
        (fx, fy, fw, fh) = faces[0]
        h, w = img.shape[:2]

        # 4. Out of Box: Face should be roughly centered
        face_cx = fx + fw // 2
        face_cy = fy + fh // 2
        margin_x = w * 0.2
        margin_y = h * 0.2
        if face_cx < margin_x or face_cx > w - margin_x or face_cy < margin_y or face_cy > h - margin_y:
            session_metrics[username]["box"] += 1
            print(f"DEBUG: Face out of box (cx={face_cx}, cy={face_cy})")
            return {"warning": "OUT_OF_BOX"}

        # 5. Eye/Gaze check using eye cascade inside the face region
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        roi_gray = gray[fy:fy+fh, fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        
        if len(eyes) >= 2:
            session_metrics[username]["gaze"] += 1
        elif len(eyes) == 0:
            # Eyes not detected — might be looking away
            print("DEBUG: Eyes not visible, possible gaze issue")
            return {"warning": "EYE_GAZE"}
        
        return {"warning": "LOOKING_GREAT"}
        
    except Exception as e:
        print(f"DEBUG: Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return {"warning": ""}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

async def background_evaluate_question(username: str, index: int, question: str, answer: str, ideal_answer: str, branch: str = "COMMON"):
    """Evaluate a single interview question in the background using Ollama AsyncClient."""
    if not answer or len(answer.strip()) < 5:
        res = {
            "accuracy": "0%",
            "score": 0.0,
            "improvement": "Providing no answer or a very brief one doesn't demonstrate your knowledge. Try to explain at least the basic concepts.",
            "status": "evaluated"
        }
        if username in session_answers and index < len(session_answers[username]):
            session_answers[username][index].update(res)
        return

    prompt = f"""
    You are an expert technical interviewer evaluating a {branch} engineering candidate. 
    Question: {question}
    Ideal Answer: {ideal_answer}
    Candidate Answer: {answer}

    Evaluation Criteria:
    - Technical Accuracy: Does the candidate understand the core concepts?
    - Completeness: Did they cover all parts of the question?
    - Clarity: Is the explanation logical and easy to follow?

    IMPORTANT:
    1. 'accuracy' should be a percentage string (e.g., "85%").
    2. 'score' should be a float from 0.0 to 10.0.
    3. 'improvement' MUST be a specific, actionable sentence for a student to improve this exact answer.

    Return ONLY JSON:
    {{
      "accuracy": "...",
      "score": ...,
      "improvement": "..."
    }}
    """
    try:
        # Use async client for non-blocking parallel execution
        client = ollama.AsyncClient()
        response = await client.generate(model='llama3', prompt=prompt, format='json')
        eval_data = json.loads(response['response'])
        eval_data["status"] = "evaluated"
        eval_data["ideal_answer"] = ideal_answer
        if username in session_answers and index < len(session_answers[username]):
            session_answers[username][index].update(eval_data)
    except Exception as e:
        print(f"Background Eval Error for {username} Q{index}: {e}")
        if username in session_answers and index < len(session_answers[username]):
            session_answers[username][index]["status"] = "error"

def analyze_voice_features(transcript: str, duration: float):
    """Calculates filler words, WPM, and generates a voice score."""
    # Handle "No response" cases to prevent 240 WPM bugs
    if not transcript or "no audible response" in transcript.lower():
        return {
            "voice_score": 0.0,
            "filler_count": 0,
            "wpm": 0,
            "feedback": "No audible response detected."
        }
        
    # Expanded filler word list for better detection
    fillers = [
        r'\bum\b', r'\buh\b', r'\berr\b', r'\bhmm\b', r'\blike\b', 
        r'\bactually\b', r'\bbasically\b', r'\byou know\b', r'\bi mean\b', r'\bso\b',
        r'\bright\b', r'\bwell\b', r'\bokay\b', r'\bouknow\b', r'\bkind of\b', r'\bsort of\b',
        r'\byou see\b', r'\banyway\b', r'\banyhow\b'
    ]
    # Use (?i) for case-insensitive matching
    filler_count = sum(len(re.findall(f, transcript.lower())) for f in fillers)
    
    words = transcript.strip().split()
    wpm = (len(words) / max(1.0, duration)) * 60
    
    # Mathematical Scoring
    filler_score = max(0.0, 10.0 - (filler_count * 2.0))
    # Target: 130-160 WPM for clear speech
    wpm_score = 10.0 if 130 <= wpm <= 160 else max(0.0, 10.0 - abs(145.0 - wpm) / 10.0)
    
    voice_score = round((filler_score * 0.6 + wpm_score * 0.4), 1)
    
    return {
        "voice_score": voice_score,
        "filler_count": filler_count,
        "wpm": round(wpm, 1),
        "feedback": f"Speech rate: {round(wpm)} WPM. {filler_count} filler words detected."
    }

async def background_process_answer(username: str, index: int, question: str, audio_path: str, ideal_answer: str, branch: str = "COMMON"):
    """Background task to handle transcription and then evaluation."""
    try:
        # 1. Transcribe (Runs in background)
        result = stt_model.transcribe(audio_path, language="en", task="transcribe")
        transcription = result.get("text", "").strip()
        
        # Get duration from whisper result segments
        duration = 0.0
        if "segments" in result and result["segments"]:
            duration = result["segments"][-1]["end"]
        
        if not transcription: transcription = "No audible response recorded."
        
        # 2. Analyze Voice Features
        voice_metrics = analyze_voice_features(transcription, duration)
        
        # 3. Update session answers with transcription and voice data
        if username in session_answers and index < len(session_answers[username]):
            session_answers[username][index].update({
                "answer": transcription,
                "status": "evaluating_technical",
                "voice_metrics": voice_metrics,
                "duration": duration
            })
        
        # 4. Trigger Evaluation (Now async/parallel)
        await background_evaluate_question(username, index, question, transcription, ideal_answer, branch)
        
        # 5. Cleanup
        if os.path.exists(audio_path): os.remove(audio_path)
    except Exception as e:
        print(f"Background Process Error for {username} Q{index}: {traceback.format_exc()}")
        if username in session_answers and index < len(session_answers[username]):
            session_answers[username][index]["status"] = "error"
        if os.path.exists(audio_path): os.remove(audio_path)

@app.post("/evaluate_step")
async def evaluate_step(
    background_tasks: BackgroundTasks,
    username: str = Form(...), 
    question: str = Form(...), 
    index: int = Form(...), 
    audio: UploadFile = File(...)
):
    username = username.strip()
    a_path = f"temp_{username}_{index}.m4a"
    
    # Fast Disk Sync
    content = await audio.read()
    with open(a_path, "wb") as f: 
        f.write(content)
    
    if username not in session_answers:
        session_answers[username] = []
    
    # Store placeholder data - Evaluation status is now "transcribing" initially
    q_data = {
        "question": question, 
        "answer": "Processing...", 
        "status": "transcribing",
        "accuracy": "0%",
        "score": 0.0,
        "improvement": "Processing audio..."
    }
    
    while len(session_answers[username]) <= index:
        session_answers[username].append(None)
    
    session_answers[username][index] = q_data

    # Fetch branch and ideal answer from DB for background eval
    db = SessionLocal()
    try:
        from models import InterviewQuestion, User
        user = db.query(User).filter(User.username == username).first()
        branch = user.branch if user else "COMMON"
        
        iq = db.query(InterviewQuestion).filter(InterviewQuestion.question == question).first()
        ideal = iq.ideal_answer if iq else "Explain the technical concepts behind this question clearly."
        
        # Trigger ALL processing in background - Transcription AND Evaluation
        background_tasks.add_task(background_process_answer, username, index, question, a_path, ideal, branch)
    finally:
        db.close()

def analyze_video_session(video_path: str):
    """Wrapper for the advanced MediaPipe-based CameraEvaluator."""
    try:
        from camera_eval import CameraEvaluator
        evaluator = CameraEvaluator()
        return evaluator.analyze_video(video_path)
    except Exception as e:
        print(f"Error in analyze_video_session: {e}")
        # Return fallback with safe metrics
        return {
            "camera_score": 0.0, 
            "camera_feedback": "Video analysis could not be completed. Please ensure your camera is working and your face is visible.",
            "metrics": {"total_frames": 0, "eye_contact": 0.0, "smile_pct": 0.0, "visibility": 0.0}
        }

@app.post("/final_session_report")
async def final_session_report(username: str = Form(...), video: Optional[UploadFile] = File(None)):
    username = username.strip()
    v_path = f"v_{username}.mp4"
    
    # Wait for all background tasks (transcription + evaluation) to finish
    qa_history = [q for q in session_answers.get(username, []) if q is not None]
    wait_start = time.time()
    while any(q.get("status") in ["transcribing", "evaluating_technical", "evaluating"] for q in qa_history) and (time.time() - wait_start < 15):
        await asyncio.sleep(0.5)

    # 1. Physical Presence & Behavioral Evaluation
    print(f"DEBUG: Processing final report for {username}. Video present: {video is not None}")
    if video:
        v_bytes = await video.read()
        print(f"DEBUG: Video received, size: {len(v_bytes)} bytes")
        with open(v_path, "wb") as f: 
            f.write(v_bytes)
        
        print(f"DEBUG: Video saved to {os.path.abspath(v_path)}. Analyzing...")
        camera_eval = analyze_video_session(v_path)
        print(f"DEBUG: Camera analysis complete. Score: {camera_eval.get('camera_score')}")
        camera_score = float(camera_eval.get("camera_score", 0.0))
        behavior_full = camera_eval # Full dictionary for DB storage
    else:
        print("DEBUG: No video provided for analysis.")
        camera_score = 0.0
        camera_eval = {"camera_score": 0.0, "camera_feedback": "No video provided.", "metrics": {}}
        behavior_full = camera_eval

    # 2. Extract Mathematical Scores for Fusion
    tech_scores = [q.get("score", 0.0) for q in qa_history if "score" in q]
    avg_tech_score = sum(tech_scores) / len(tech_scores) if tech_scores else 0.0
    
    voice_scores = [q.get("voice_metrics", {}).get("voice_score", 0.0) for q in qa_history if "voice_metrics" in q]
    avg_voice_score = sum(voice_scores) / len(voice_scores) if voice_scores else 0.0
    
    # 3. Final Fusion Formula: 50% Technical + 25% Voice/Communication + 25% Camera/Behavior
    avg_tech_score = round(avg_tech_score, 1)
    avg_voice_score = round(avg_voice_score, 1)
    camera_score = round(camera_score, 1)
    fusion_score = round((avg_tech_score * 0.5) + (avg_voice_score * 0.25) + (camera_score * 0.25), 1)
    
    # 4. Integrate Live Warning History (Audit from frame analysis)
    # This keeps the Haar Cascade warnings about lighting/cheating/etc as requested.
    live_metrics = session_metrics.pop(username, {"frames": 0, "face": 0, "gaze": 0, "smile": 0, "light": 0, "multi": 0, "box": 0})
    warning_notes = []
    if live_metrics["light"] > 10: warning_notes.append("Issues with low lighting detected.")
    if live_metrics["multi"] > 0: warning_notes.append("Multiple people detected during session.")
    if live_metrics["box"] > 10: warning_notes.append("Candidate was frequently out of camera frame.")
    
    qa_text = ""
    session_transcript = ""
    total_duration = 0.0
    for item in qa_history:
        ans = item.get('answer', "")
        # Filter out placeholders to avoid counting them as speech
        if ans and ans not in ["No response recorded.", "Processing...", "No audible response recorded.", "No speech detected."]:
            session_transcript += ans + " "
            total_duration += item.get("duration", 0.0)
        
        v_m = item.get("voice_metrics", {})
        qa_text += f"Question: {item['question']}\nTechnical Accuracy: {item.get('accuracy', '0%')}\nCommunication Details: {v_m.get('feedback', 'N/A')}\n\n"

    # 4b. Global Speech Confidence Calculation
    session_confidence = calculate_confidence(session_transcript, total_duration)
    
    # Generate brief inference for the student
    if session_confidence['word_count'] == 0:
        confidence_inference = "No speech was detected during this session. Please ensure your microphone is working and you are speaking clearly."
    else:
        cp = session_confidence['analysis']
        inference_parts = []
        if cp['pace'] == "Good":
            inference_parts.append("Your speaking pace is professional and easy to follow.")
        else:
            inference_parts.append(f"Your speaking pace {cp['pace'].lower()}, consider adjusting it for better clarity.")
            
        if cp['fillers'] == "Low":
            inference_parts.append("Great job avoiding filler words, you sound very confident.")
        elif cp['fillers'] == "Noticeable":
            inference_parts.append("You used some filler words; pausing briefly can help reduce them.")
        else:
            inference_parts.append("High filler word usage detected. Practice structured thinking to minimize hesitation.")
            
        if cp['length'] == "Short":
            inference_parts.append("Your answers were a bit brief. Try providing more context or examples in technical discussions.")
            
        confidence_inference = " ".join(inference_parts)

    # Prepare detailed voice metrics for the prompt
    total_fillers = sum(q.get("voice_metrics", {}).get("filler_count", 0) for q in qa_history)
    avg_wpm = sum(q.get("voice_metrics", {}).get("wpm", 0.0) for q in qa_history) / len(qa_history) if qa_history else 0.0
    
    # Camera metrics for prompt
    cam_m = camera_eval.get("metrics", {})
    eye_pct = cam_m.get("eye_contact", 0.0)
    smile_pct = cam_m.get("smile_pct", 0.0)

    prompt = f"""
    TECHNICAL: {avg_tech_score}/10
    VOICE: {avg_voice_score}/10 (Fillers: {total_fillers}, Avg WPM: {round(avg_wpm)})
    CAMERA: {camera_score}/10 (Eye Contact: {eye_pct}%, Smiles: {smile_pct}%)
    FUSION: {fusion_score}/10
    
    AUDIT: {", ".join(warning_notes) if warning_notes else "Clean."}
    
    TRANSCRIPTS:
    {qa_text}

    Generate a PROFESSIONAL JSON report. 
    Include 'filler_words_count': {total_fillers}.
    Explicitly detail Voice (speed, fillers) and Camera (eye contact, smiles) in 'behavioral_feedback'.
    Ensure 'final_score' is "{fusion_score}/10".
    
    Format JSON:
    {{
      "final_score": "{fusion_score}/10",
      "individual_scores": {{
          "technical": "{avg_tech_score}/10",
          "voice": "{avg_voice_score}/10",
          "camera": "{camera_score}/10"
      }},
      "overall_confidence": "Concise summary...",
      "behavioral_feedback": "A professional narrative describing communication and behavior...",
      "technical_report": [
         {{
           "question": "...",
           "your_answer": "...",
           "accuracy": "XX%",
           "ideal_answer": "...",
           "improvement": "...",
           "voice_stats": {{
              "wpm": 0,
              "fillers": 0,
              "feedback": "..."
           }}
         }}
      ],
      "session_metrics": {{
         "eye_contact": {eye_pct},
         "smiles": {smile_pct},
         "avg_wpm": {round(avg_wpm)},
         "total_fillers": {total_fillers}
      }},
      "speech_confidence": {{
         "score": {session_confidence['score']},
         "analysis": {json.dumps(session_confidence['analysis'])},
         "inference": "{confidence_inference}"
      }}
    }}
    """
    
    try:
        response = ollama.generate(model='llama3', prompt=prompt, format='json')
        report = json.loads(response['response'])
        
        # Enforce string types for Flutter stability - CRITICAL: Prevent raw dicts
        for key in ["overall_confidence", "behavioral_feedback", "final_score"]:
            val = report.get(key)
            if not isinstance(val, str):
                report[key] = str(val) if val else "N/A"

        # Explicitly inject metrics for frontend
        report["filler_words_count"] = total_fillers
        report["individual_scores"] = {
            "technical": f"{avg_tech_score}/10",
            "voice": f"{avg_voice_score}/10",
            "camera": f"{camera_score}/10"
        }

        # Sync pre-evaluated details
        for idx, q_report in enumerate(report.get("technical_report", [])):
            if idx < len(qa_history):
                q_report["improvement"] = str(qa_history[idx].get("improvement", q_report.get("improvement", "N/A")))
                q_report["accuracy"] = str(qa_history[idx].get("accuracy", "0%"))
                q_report["your_answer"] = str(qa_history[idx].get("answer", ""))
                q_report["ideal_answer"] = str(qa_history[idx].get("ideal_answer", q_report.get("ideal_answer", "N/A")))
                
                # Inject Voice Metrics per question
                v_m = qa_history[idx].get("voice_metrics", {})
                q_report["voice_stats"] = {
                    "wpm": v_m.get("wpm", 0),
                    "fillers": v_m.get("filler_count", 0),
                    "feedback": v_m.get("feedback", "N/A")
                }
    except Exception as e:
        print(f"Ollama Final Report Error: {e}")
        report = {
            "final_score": f"{fusion_score}/10",
            "overall_confidence": "Technical session completed.",
            "behavioral_feedback": f"Voice Score: {avg_voice_score}. Camera Score: {camera_score}.",
            "session_metrics": {
                "eye_contact": eye_pct,
                "smiles": smile_pct,
                "avg_wpm": round(avg_wpm),
                "total_fillers": total_fillers
            },
            "technical_report": [
                {
                    "question": q['question'],
                    "your_answer": q.get('answer', ""),
                    "accuracy": q.get('accuracy', "0%"),
                    "ideal_answer": "Refer to dashboard.",
                    "improvement": q.get('improvement', "Practice."),
                    "voice_stats": {
                        "wpm": q.get("voice_metrics", {}).get("wpm", 0),
                        "fillers": q.get("voice_metrics", {}).get("filler_count", 0),
                        "feedback": q.get("voice_metrics", {}).get("feedback", "N/A")
                    }
                } for q in qa_history
            ]
        }
    
    # 5. Persistent DB Storage
    final_report = dict(report)
    final_report["content_score"] = avg_tech_score
    final_report["camera_score"] = camera_score
    final_report["voice_score"] = avg_voice_score
    final_report["final_score"] = fusion_score
    final_report["filler_words_count"] = total_fillers
    final_report["eye_contact_percent"] = eye_pct
    final_report["smile_percent"] = smile_pct

    # Ensure individual_scores exists for DB retrieval stability
    final_report["individual_scores"] = {
        "technical": f"{avg_tech_score}/10",
        "voice": f"{avg_voice_score}/10",
        "camera": f"{camera_score}/10"
    }
    
    # EXPLICITLY STORE TRANSCRIPTS in the report if AI missed them
    for idx, detail in enumerate(final_report.get("technical_report", [])):
        if idx < len(qa_history):
            # Prioritize the actual recorded answer
            detail["your_answer"] = qa_history[idx].get("answer", detail.get("your_answer", ""))
            detail["user_input"] = detail["your_answer"]
            # Ensure voice_stats is preserved
            if "voice_stats" not in detail:
                 v_m = qa_history[idx].get("voice_metrics", {})
                 detail["voice_stats"] = {
                    "wpm": v_m.get("wpm", 0),
                    "fillers": v_m.get("filler_count", 0),
                    "feedback": v_m.get("feedback", "N/A")
                }

    db = SessionLocal()
    try:
        from models import Score, InterviewDetail
        # Store detailed breakdown in confidence field for history retrieval
        confidence_data = {
            "vision": behavior_full, 
            "voice": {
                "avg_score": avg_voice_score,
                "details": [q.get("voice_metrics", {}) for q in qa_history]
            }, 
            "warnings": warning_notes,
            "scores": {
                "tech": avg_tech_score,
                "voice": avg_voice_score,
                "camera": camera_score
            }
        }
        
        new_result = Score(
            username=username,
            category="INTERVIEW",
            score=fusion_score,
            area="Confidence Evaluation",
            confidence=json.dumps(confidence_data),
            total_questions=10,
            timestamp=datetime.now(timezone.utc)
        )
        db.add(new_result)
        db.commit()
        db.refresh(new_result)
        res_id = new_result.id

        for detail in final_report.get("technical_report", []):
            db.add(InterviewDetail(
                result_id=res_id,
                question=detail.get("question", ""),
                user_answer=detail.get("your_answer", ""),
                ideal_answer=detail.get("ideal_answer", ""),
                improvement=detail.get("improvement", ""),
                accuracy=str(detail.get("accuracy", "0%"))
            ))
        db.commit()
    except Exception as e:
        print(f"DB WRITE ERROR: {e}")
        db.rollback()
    finally:
        db.close()

    if os.path.exists(v_path): os.remove(v_path)
    session_answers[username] = [] 
    return final_report


@app.get("/history/{username}", tags=["Analytics"])
async def get_history(username: str, db: Session = Depends(get_db)):
    """Get all session history for a user"""
    try:
        res = db.execute(text("""
            SELECT id, category, DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE), score, total_questions, area 
            FROM results 
            WHERE username = :u 
            ORDER BY timestamp DESC LIMIT 50
        """), {"u": username}).fetchall()
        
        history = []
        for r in res:
            history.append({
                "id": r[0],
                "category": r[1],
                "date": str(r[2]),
                "score": float(r[3]),
                "total": r[4],
                "area": r[5]
            })
        return history
    except Exception as e:
        print(f"HISTORY ERROR: {e}")
        return []

@app.get("/performance_by_date/{username}/{target_date}", tags=["Analytics"])
async def get_performance_by_date(username: str, target_date: str, db: Session = Depends(get_db)):
    """Get performance data for a specific date (week view)"""
    try:
        # Calculate Monday of that target_date
        target = datetime.strptime(target_date, "%Y-%m-%d")
        monday = target - timedelta(days=target.weekday())
        sunday = monday + timedelta(days=6)

        performance = {"aptitude": [], "technical": [], "interview": [], "gd": []}
        for cat in ["APTITUDE", "TECHNICAL", "INTERVIEW", "GD"]:
            res = db.execute(text("""
                SELECT DATE(timestamp), MAX(score) 
                FROM results 
                WHERE username = :u AND category = :cat 
                AND DATE(timestamp) BETWEEN :start AND :end
                GROUP BY DATE(timestamp)
            """), {"u": username, "cat": cat, "start": monday, "end": sunday}).fetchall()
            
            performance[cat.lower()] = [{"day": str(r[0]), "score": float(r[1])} for r in res]
            # Fixing the above line: performance[cat.lower()] = [{"day": str(row[0]), "score": float(row[1])} for row in res]

        return {"performance": performance}
    except Exception as e:
        print(f"PERFORMANCE BY DATE ERROR: {e}")
        return {"performance": {}}

@app.get("/session_detail/{category}/{session_id}", tags=["Analytics"])
async def get_session_detail(category: str, session_id: int, db: Session = Depends(get_db)):
    """Get detailed breakdown for a specific session"""
    try:
        cat = category.upper()
        if cat == "GD":
            from models import GDResult
            # First try finding by result_id (new system)
            res = db.query(GDResult).filter(GDResult.result_id == session_id).first()
            # Fallback to finding by GDResult's own primary key (old system)
            if not res:
                res = db.query(GDResult).filter(GDResult.id == session_id).first()
            
            if not res: return {"error": "Not found"}
            # Fetch real topic text — try gd_topics_extra first (new), fallback to gd_topics
            topic_text = "GD Topic"
            try:
                topic_row = db.execute(
                    text("SELECT question FROM gd_topics_extra WHERE question_id = :tid"),
                    {"tid": res.topic_id}
                ).fetchone()
                if topic_row:
                    topic_text = topic_row[0]
                else:
                    topic_row = db.execute(
                        text("SELECT topic FROM gd_topics WHERE id = :tid"),
                        {"tid": res.topic_id}
                    ).fetchone()
                    if topic_row:
                        topic_text = topic_row[0]
            except Exception:
                pass
            return {
                "topic": topic_text,
                "transcript": res.user_answer,
                "feedback": res.feedback,
                "ideal_answer": res.ideal_answer,
                "scores": {
                    "content": res.content_score,
                    "communication": res.communication_score,
                    "camera": res.camera_score
                }
            }
        elif cat == "INTERVIEW":
            from models import Score, InterviewDetail
            score = db.query(Score).filter(Score.id == session_id).first()
            details = db.query(InterviewDetail).filter(InterviewDetail.result_id == session_id).all()
            
            behavioral = {}
            try:
                behavioral = json.loads(score.confidence) if score and score.confidence else {}
            except: pass

            return {
                "area": score.area if score else "Interview",
                "behavioral_report": behavioral,
                "scores": behavioral.get("scores", {
                    "tech": score.score if score else 0.0,
                    "voice": behavioral.get("voice", {}).get("avg_score", 0.0),
                    "camera": behavioral.get("vision", {}).get("camera_score", 0.0)
                }),
                "technical_report": [
                    {
                        "question": d.question,
                        "your_answer": d.user_answer,
                        "ideal_answer": d.ideal_answer,
                        "improvement": d.improvement,
                        "accuracy": d.accuracy
                    } for d in details
                ]
            }
        else: # QUIZ (APTITUDE/TECHNICAL)
            from models import Score, QuizAnswer, Question
            score = db.query(Score).filter(Score.id == session_id).first()
            answers = db.query(QuizAnswer).filter(QuizAnswer.result_id == session_id).all()
            
            detailed_qs = []
            for ans in answers:
                q = db.query(Question).filter(Question.id == ans.question_id).first()
                if q:
                    # Return None if user didn't select anything (empty string or None)
                    user_pick = ans.user_answer if ans.user_answer and ans.user_answer.strip() else None
                    is_skipped = user_pick is None
                    detailed_qs.append({
                        "question": q.question,
                        "user_selected": user_pick,
                        "is_skipped": is_skipped,
                        "correct_answer": q.correct_answer,
                        "is_correct": bool(ans.is_correct) and not is_skipped,
                        "explanation": q.explanation,
                        "options": [q.option_a, q.option_b, q.option_c, q.option_d]
                    })
            
            return {
                "area": score.area if score else "General",
                "score": score.score if score else 0.0,
                "scores": {"total": score.score if score else 0.0},
                "questions": detailed_qs
            }
    except Exception as e:
        print(f"SESSION DETAIL ERROR: {e}")
        return {"error": str(e)}

@app.get("/suggestions/{username}", tags=["Student Insights"])
def get_user_suggestions(username: str, db: Session = Depends(get_db)):
    """Get all suggestions sent to this student by teachers."""
    try:
        from models import TeacherSuggestion
        suggestions = db.query(TeacherSuggestion).filter(
            TeacherSuggestion.student_username == username
        ).order_by(TeacherSuggestion.timestamp.desc()).all()
        
        return {
            "status": "success",
            "suggestions": [
                {
                    "id": s.id,
                    "teacher": s.teacher_username,
                    "message": s.message,
                    "is_read": bool(s.is_read),
                    "timestamp": s.timestamp.isoformat() if s.timestamp else None
                }
                for s in suggestions
            ]
        }
    except Exception as e:
        print(f"GET SUGGESTIONS ERROR: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/suggestions/{suggestion_id}/read", tags=["Student Insights"])
def mark_suggestion_as_read(suggestion_id: int, db: Session = Depends(get_db)):
    """Mark a specific teacher suggestion as read."""
    try:
        from models import TeacherSuggestion
        suggestion = db.query(TeacherSuggestion).filter(TeacherSuggestion.id == suggestion_id).first()
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        
        suggestion.is_read = 1
        db.commit()
        return {"status": "success", "message": "Suggestion marked as read"}
    except Exception as e:
        db.rollback()
        print(f"MARK SUGGESTION READ ERROR: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/", tags=["Root"])
def root():
    return {
        "app": "Placement Preparation Platform",
        "version": "4.0.0",
        "database": "MySQL",
        "features": {
            "adaptive_quiz": "10 questions daily with progressive difficulty",
            "answer_validation": "Fixed - accurate checking",
            "level_system": "Auto-level up at 70% score",
            "interview_practice": "Voice-based AI evaluation",
            "analytics": "Performance tracking"
        },
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
