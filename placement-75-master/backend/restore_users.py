from database import mysql_engine, Base
from models import User
from sqlalchemy import text

def restore_users_table():
    print("Attempting to restore 'users' table...")
    try:
        # Check if table exists first
        with mysql_engine.connect() as conn:
            result = conn.execute(text("SHOW TABLES LIKE 'users'"))
            if result.fetchone():
                print("Table 'users' already exists. No action taken.")
                return

        # Recreate the users table using the model definition
        User.__table__.create(mysql_engine)
        print("Successfully recreated 'users' table.")
        
        # Verify
        with mysql_engine.connect() as conn:
            result = conn.execute(text("SHOW TABLES LIKE 'users'"))
            if result.fetchone():
                print("Verification: 'users' table is now present in the database.")
            else:
                print("Verification FAILED: 'users' table still missing after creation attempt.")
                
    except Exception as e:
        print(f"Error during restoration: {e}")

if __name__ == "__main__":
    restore_users_table()
