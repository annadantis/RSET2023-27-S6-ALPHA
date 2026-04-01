import json
import ollama
from database import SessionLocal, init_db
from models import InterviewQuestion

def generate_questions_via_ai(branch: str, count: int = 10) -> list:
    """Generates a batch of interview questions purely via Ollama AI."""
    print(f"Generating {count} questions for {branch} purely via AI...")
    
    prompt = f"""
    You are an expert HR and technical interviewer. 
    Provide exactly {count} high-quality interview questions for a fresh graduate in the '{branch}' engineering branch.
    (If branch is 'COMMON', provide general HR/Behavioral questions).
    
    Provide the output as a JSON array where each element is an object with:
    - "question": The interview question.
    - "ideal_answer": A concise (1-2 sentences) ideal answer to look out for.
    
    Return ONLY valid JSON format.
    """
    
    questions = []
    try:
        response = ollama.generate(model='llama3', prompt=prompt, format='json')
        questions_json = json.loads(response['response'])
        
        if isinstance(questions_json, dict) and len(questions_json.keys()) >= 1:
            questions_json = questions_json[list(questions_json.keys())[0]]
            
        if isinstance(questions_json, list):
            for q_obj in questions_json:
                if q_obj.get("question") and q_obj.get("ideal_answer"):
                     questions.append({
                         "question": q_obj.get("question", ""),
                         "ideal_answer": q_obj.get("ideal_answer", "")
                     })
            print(f"Successfully generated {len(questions)} questions for {branch}.")
        else:
            print(f"Unexpected JSON format returned by AI for {branch}.")
    except Exception as e:
        print(f"Failed to generate questions via AI for {branch}: {e}")
        
    return questions

def seed_database():
    print("Ensuring database tables exist...")
    init_db()
    
    db = SessionLocal()
    
    try:
        # Target branches
        target_branches = ["CSE", "IT", "AIDS", "CSBS", "EEE", "ECE", "AEI", "MECH", "CIVIL", "COMMON"]
        
        print("Checking current database state...")
        for branch in target_branches:
            # Check how many questions we currently have for this branch
            count = db.query(InterviewQuestion).filter(InterviewQuestion.branch == branch).count()
            
            # If we already have a healthy amount, skip to save time. 
            # (Background tasks will handle replenishing low branches dynamically)
            if count >= 10:
                print(f"  {branch}: Already has {count} questions. Skipping.")
                continue
                
            print(f"  {branch}: Only has {count} questions. Generating more...")
            
            # We want at least 10 in the database to start
            needed = 10 - count
            new_qs = generate_questions_via_ai(branch, needed)
            
            for q_data in new_qs:
                 new_record = InterviewQuestion(
                      branch=branch,
                      question=q_data["question"],
                      ideal_answer=q_data["ideal_answer"]
                 )
                 db.add(new_record)
                 
            db.commit()
            print(f"  -> Added {len(new_qs)} new questions for {branch}.\n")
            
        print("Database seeding/verification complete.")

    except Exception as e:
        db.rollback()
        print(f"Error during seeding process: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
