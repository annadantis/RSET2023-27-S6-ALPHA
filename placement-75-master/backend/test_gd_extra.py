from database import mysql_engine as engine
from sqlalchemy import text
import traceback

def check_gd_topics_extra():
    print("--- DESCRIBING gd_topics_extra ---")
    try:
        with engine.connect() as con:
            res = con.execute(text("DESCRIBE gd_topics_extra"))
            rows = res.fetchall()
            for row in rows:
                print(str(row))
    except Exception as e:
        print(f"Error DESCRIBE gd_topics_extra: {e}")

    print("--- FETCHING FROM gd_topics_extra ---")
    try:
        with engine.connect() as con:
            res = con.execute(text("SELECT id, topic, keywords FROM gd_topics_extra ORDER BY RAND() LIMIT 1"))
            rows = res.fetchall()
            for row in rows:
                print(str(row))
    except Exception as e:
        print(f"Error SELECT gd_topics_extra: {e}")

if __name__ == "__main__":
    check_gd_topics_extra()