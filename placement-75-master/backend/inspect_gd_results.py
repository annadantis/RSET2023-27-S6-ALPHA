from database import mysql_engine as engine
from sqlalchemy import text

def inspect():
    with engine.connect() as con:
        print("Inspecting gd_results table...")
        res = con.execute(text("DESCRIBE gd_results")).fetchall()
        for row in res:
            print(row)

if __name__ == "__main__":
    inspect()
