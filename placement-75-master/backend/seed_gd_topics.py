import sys
import os
from sqlalchemy import text
from database import SessionLocal, init_db

def seed_topics():
    # Ensure tables exist
    init_db()
    
    db = SessionLocal()
    try:
        # Check if topics already exist
        count = db.execute(text("SELECT COUNT(*) FROM gd_topics")).scalar()
        if count > 0:
            print(f"✅ Database already contains {count} topics. Skipping seed.")
            return

        topics = [
            "Impact of Artificial Intelligence on Job Market",
            "Pros and Cons of Remote Work Culture",
            "Social Media: A Boon or a Bane for Mental Health?",
            "Is Universal Basic Income the solution to poverty?",
            "The Future of Electric Vehicles in India",
            "Cryptocurrency: The future of money or a bubble?",
            "Climate Change: Individual Responsibility vs Corporate Responsibility",
            "The Role of Ethics in Business and Corporate Governance",
            "Online Education vs traditional Classroom Learning",
            "Cybersecurity: Challenges in the Digital Age"
        ]

        print("🌱 Seeding GD topics...")
        for topic in topics:
            db.execute(
                text("INSERT INTO gd_topics (topic) VALUES (:topic)"),
                {"topic": topic}
            )
        
        db.commit()
        print(f"✅ Successfully seeded {len(topics)} topics.")
    except Exception as e:
        print(f"❌ Error seeding topics: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_topics()
