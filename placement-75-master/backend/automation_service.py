from database import SessionLocal
from question_generator import generate_questions_ai
from sqlalchemy import text
from datetime import date, datetime

def daily_question_job():
    """
    Job to ensure questions are balanced and exist for all users.
    Triggered manually or via scheduler.
    """
    print(f"🚀 [Automation] Running daily question generation job at {datetime.now()}")
    # Logic to replenish questions could go here
    return True
