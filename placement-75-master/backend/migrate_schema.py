from database import mysql_engine as engine
from sqlalchemy import text

def migrate():
    print("Starting Robust Database Migration...")
    
    with engine.connect() as con:
        # 1. Update gd_topics
        try:
            con.execute(text("ALTER TABLE gd_topics ADD COLUMN keywords TEXT AFTER topic"))
            con.commit()
            print("Added keywords to gd_topics")
        except Exception as e:
            if "Duplicate column" in str(e): print("keywords already exists in gd_topics")
            else: print(f"Error adding keywords to gd_topics: {e}")

        # 2. Update gd_results
        cols_to_add = [
            ("username", "VARCHAR(255) AFTER topic_id"),
            ("voice_score", "FLOAT AFTER camera_score"),
            ("overall_score", "FLOAT AFTER final_score"),
            ("content_audit", "TEXT AFTER overall_score"),
            ("found_keywords", "TEXT AFTER content_audit"),
            ("missing_keywords", "TEXT AFTER found_keywords"),
            ("improved_answer", "TEXT AFTER missing_keywords"),
            ("strategy_note", "TEXT AFTER improved_answer"),
            ("result_id", "INT AFTER ideal_answer")
        ]

        for col, definition in cols_to_add:
            try:
                con.execute(text(f"ALTER TABLE gd_results ADD COLUMN {col} {definition}"))
                con.commit()
                print(f"Added {col} to gd_results")
            except Exception as e:
                if "Duplicate column" in str(e): print(f"{col} already exists in gd_results")
                else: print(f"Error adding {col} to gd_results: {e}")

        # 3. Modify types
        mods = [
            "MODIFY COLUMN content_score FLOAT",
            "MODIFY COLUMN communication_score FLOAT",
            "MODIFY COLUMN camera_score FLOAT",
            "MODIFY COLUMN final_score FLOAT",
            "MODIFY COLUMN topic_id VARCHAR(50)"
        ]
        for mod in mods:
            try:
                con.execute(text(f"ALTER TABLE gd_results {mod}"))
                con.commit()
                print(f"Applied mod: {mod}")
            except Exception as e:
                print(f"Error applying mod {mod}: {e}")

    print("Migration complete.")

    print("Migration complete.")

if __name__ == "__main__":
    migrate()


