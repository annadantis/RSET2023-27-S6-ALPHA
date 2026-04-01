
from backend.database import SessionLocal
from sqlalchemy import text
import json

def check_db():
    db = SessionLocal()
    try:
        # Check timezone and current time
        tz_info = db.execute(text("SELECT @@session.time_zone, @@global.time_zone, NOW()")).fetchone()
        print(f"Session TZ: {tz_info[0]}, Global TZ: {tz_info[1]}, DB Now: {tz_info[2]}")
        
        # Check if any interviews exist
        interviews = db.execute(text("SELECT id, username, category, score, timestamp FROM results WHERE category='INTERVIEW' ORDER BY timestamp DESC LIMIT 10")).fetchall()
        print(f"\nLast 10 Interviews:")
        for i in interviews:
            print(f"ID: {i[0]}, User: {i[1]}, Cat: {i[2]}, Score: {i[3]}, Timestamp: {i[4]}")
            
        # Check today's interviews with the logic used in daily_report
        # Let's see what the DB thinks 'today' is in IST
        ist_date_query = db.execute(text("SELECT DATE(DATE_ADD(NOW(), INTERVAL '5:30' HOUR_MINUTE))")).fetchone()
        print(f"\nDB thinks today (IST) is: {ist_date_query[0]}")
        
        target_date = ist_date_query[0]
        
        report_query = db.execute(
            text("""
                SELECT id, category, timestamp 
                FROM results 
                WHERE category='INTERVIEW' AND DATE(DATE_ADD(timestamp, INTERVAL '5:30' HOUR_MINUTE)) = :d
            """),
            {"d": target_date}
        ).fetchall()
        
        print(f"\nInterviews found for {target_date} using report logic: {len(report_query)}")
        for r in report_query:
            print(f"ID: {r[0]}, Cat: {r[1]}, Timestamp: {r[2]}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    check_db()
