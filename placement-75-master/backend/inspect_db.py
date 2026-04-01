from sqlalchemy import text
from database import engine

def inspect_columns(table_name):
    print(f"\nColumns in {table_name}:")
    try:
        with engine.connect() as connection:
            result = connection.execute(text(f"DESCRIBE {table_name}"))
            for row in result:
                print(row)
    except Exception as e:
        print(f"Error inspecting {table_name}: {e}")

if __name__ == "__main__":
    inspect_columns("questions")
    inspect_columns("results")
    inspect_columns("gd_results")
    inspect_columns("users")
    inspect_columns("gd_topics")
    inspect_columns("interview_questions")
