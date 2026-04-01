import os
import sys
import pymysql
from sqlalchemy import create_engine, text
from database import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

def get_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=int(DB_PORT),
        cursorclass=pymysql.cursors.DictCursor
    )

def resequence_ids():
    """Exact logic from resequence_ids.py - resets IDs to start from 1."""
    print("Resequencing Question IDs...")
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM questions ORDER BY id ASC")
            questions = cursor.fetchall()
            if not questions: return

            cursor.execute("TRUNCATE TABLE questions")
            columns = [k for k in questions[0].keys() if k != 'id']
            sql = f"INSERT INTO questions ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})"
            values = [[q[col] for col in columns] for q in questions]
            cursor.executemany(sql, values)
            conn.commit()
            print("Successfully re-sequenced IDs.")
    except Exception as e: print(f"Error: {e}")
    finally: conn.close()

def sync_branches():
    """Exact logic from update_branch_values.py - ensures correct branch tagging."""
    print("Syncing question branch values...")
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Aptitude/GD -> Common
            cursor.execute("""
                UPDATE questions SET branch = 'Common' 
                WHERE category IN ('aptitude', 'general aptitude', 'gd', 'group discussion')
            """)
            # Technical -> CSE (if blank)
            cursor.execute("""
                UPDATE questions SET branch = 'CSE' 
                WHERE category = 'technical' AND (branch IS NULL OR branch = '' OR branch = 'generated')
            """)
            conn.commit()
            print("Branch sync complete.")
    except Exception as e: print(f"Error: {e}")
    finally: conn.close()

def update_schema():
    """Handles common column additions and modifications."""
    print("Verifying database schema...")
    conn = get_connection()
    queries = [
        "ALTER TABLE questions ADD COLUMN IF NOT EXISTS option_e TEXT",
        "ALTER TABLE questions ADD COLUMN IF NOT EXISTS branch VARCHAR(50)",
        "ALTER TABLE questions ADD COLUMN IF NOT EXISTS difficulty_level INT DEFAULT 1",
        "ALTER TABLE questions ADD COLUMN IF NOT EXISTS area VARCHAR(100)",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(20) DEFAULT 'student'",
        "ALTER TABLE questions MODIFY COLUMN explanation TEXT"
    ]
    try:
        with conn.cursor() as cursor:
            for q in queries:
                try: cursor.execute(q)
                except: pass
            conn.commit()
            print("Schema updates complete.")
    except Exception as e: print(f"Error: {e}")
    finally: conn.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--resequence": resequence_ids()
        elif sys.argv[1] == "--sync": sync_branches()
        elif sys.argv[1] == "--schema": update_schema()
    else:
        print("Usage: python db_maintenance.py [--resequence | --sync | --schema]")
