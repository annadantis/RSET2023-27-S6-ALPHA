import os
import sys
from sqlalchemy import text
from database import mysql_engine as engine

def check_health():
    """Consolidated logic from check_db.py and check_ollama_health.py."""
    print("--- System Health Check ---")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("✅ Database: Connected")
            
            q_count = conn.execute(text("SELECT COUNT(*) FROM questions")).scalar()
            u_count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
            print(f"✅ Data Stats: {q_count} Questions, {u_count} Users")
    except Exception as e:
        print(f"❌ Database Error: {e}")

    try:
        import ollama
        ollama.list()
        print("✅ Ollama: Online and reachable")
    except:
        print("⚠️ Ollama: Local service not detected")

def analyze_distribution():
    """Consolidated logic from check_question_distribution.py and analyze_tech_dist.py."""
    print("\n--- Question Distribution ---")
    with engine.connect() as conn:
        # Branch Dist
        print("\n[By Branch]")
        res = conn.execute(text("SELECT branch, COUNT(*) FROM questions GROUP BY branch"))
        for row in res: print(f"  {row[0] or 'N/A'}: {row[1]}")
        
        # Category Dist
        print("\n[By Category]")
        res = conn.execute(text("SELECT category, COUNT(*) FROM questions GROUP BY category"))
        for row in res: print(f"  {row[0]}: {row[1]}")
        
        # Missing Explanations
        missing = conn.execute(text("SELECT COUNT(*) FROM questions WHERE explanation IS NULL OR explanation = ''")).scalar()
        print(f"\n⚠️ Questions missing explanations: {missing}")

def audit_questions():
    """Logic from audit_database.py and check_data_quality.py."""
    print("\n--- Data Quality Audit ---")
    with engine.connect() as conn:
        # Check for placeholder explanations
        placeholders = conn.execute(text("SELECT COUNT(*) FROM questions WHERE explanation LIKE '%The correct answer is%'")).scalar()
        print(f"  Placeholder explanations: {placeholders}")
        
        # Check for branch consistency
        bad_branch = conn.execute(text("SELECT COUNT(*) FROM questions WHERE branch IS NULL OR branch = ''")).scalar()
        print(f"  Unassigned branches: {bad_branch}")

if __name__ == "__main__":
    check_health()
    analyze_distribution()
    audit_questions()
