from sqlalchemy import Column, Integer, Float, String, Text, TIMESTAMP, ForeignKey
from sqlalchemy.sql import func
from database import Base

class User(Base):
    """User model for authentication and progress tracking"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=True)
    password_hash = Column(String(255), nullable=False)
    aptitude_level = Column(Integer, default=1)
    technical_level = Column(Integer, default=1)
    branch = Column(String(50))
    role = Column(String(20), default='student')
    last_level_update = Column(TIMESTAMP, server_default=func.now())
    created_at = Column(TIMESTAMP, server_default=func.now())


class Score(Base):
    """Score/Results model for tracking quiz performance"""
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), nullable=False)
    category = Column(String(50), nullable=False)
    score = Column(Float, nullable=False)
    total_questions = Column(Integer, default=10)
    area = Column(String(100))
    confidence = Column(Text)
    timestamp = Column(TIMESTAMP, server_default=func.now())


class Question(Base):
    """Question model for storing quiz questions"""
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    option_a = Column(Text)
    option_b = Column(Text)
    option_c = Column(Text)
    option_d = Column(Text)
    correct_answer = Column(Text)
    category = Column(String(50))
    area = Column(String(100))
    difficulty = Column(String(20))
    explanation = Column(Text)
    branch = Column(String(50))
    difficulty_level = Column(Integer, default=1)
    created_at = Column(TIMESTAMP, server_default=func.now())


class GDEvaluation(Base):
    """Model for storing Group Discussion evaluations"""
    __tablename__ = "gd_evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255))
    topic = Column(Text)
    transcript = Column(Text)
    content_score = Column(Integer)
    communication_score = Column(Integer)
    feedback = Column(Text)
    audio_path = Column(String(500))
    timestamp = Column(TIMESTAMP, server_default=func.now())


class WeeklyStat(Base):
    """Model for tracking historical weekly performance"""
    __tablename__ = "weekly_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), nullable=False)
    week_start_date = Column(TIMESTAMP, server_default=func.now())
    avg_score = Column(Float)  # Aggregate score across all modules
    is_level_up = Column(Integer, default=0)
    total_activities = Column(Integer, default=0)


class GDTopic(Base):
    """Model for storing Group Discussion topics"""
    __tablename__ = "gd_topics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic = Column(Text, nullable=False)
    keywords = Column(Text)  # List of industry-specific terms
    created_at = Column(TIMESTAMP, server_default=func.now())


class GDResult(Base):
    """Model for storing Group Discussion results and evaluations"""
    __tablename__ = "gd_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic_id = Column(Integer, nullable=False)
    username = Column(String(255))
    user_answer = Column(Text)
    content_score = Column(Float)
    communication_score = Column(Float)
    camera_score = Column(Float)
    voice_score = Column(Float)
    final_score = Column(Float)
    overall_score = Column(Float)
    content_audit = Column(Text)
    found_keywords = Column(Text)
    missing_keywords = Column(Text)
    improved_answer = Column(Text)
    strategy_note = Column(Text)
    feedback = Column(Text)
    ideal_answer = Column(Text)
    result_id = Column(Integer)  # Link to results.id
    timestamp = Column(TIMESTAMP, server_default=func.now())


class InterviewDetail(Base):
    """Model for storing detailed interview question breakdown"""
    __tablename__ = "interview_details"

    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(Integer, nullable=False)  # Link to results.id
    question = Column(Text)
    user_answer = Column(Text)
    accuracy = Column(String(50))
    ideal_answer = Column(Text)
    improvement = Column(Text)


class QuizAnswer(Base):
    """Model for storing per-question breakdown for quizzes"""
    __tablename__ = "quiz_answers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(Integer, nullable=False)  # Link to results.id
    question_id = Column(Integer, nullable=False)
    user_answer = Column(String(10))
    is_correct = Column(Integer)  # 1 for True, 0 for False

class InterviewQuestion(Base):
    """Model for storing generated interview questions"""
    __tablename__ = "interview_questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    branch = Column(String(50), nullable=False)
    question = Column(Text, nullable=False)
    ideal_answer = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())

class UserAskedQuestion(Base):
    """Model for tracking questions asked to a user to avoid repeats"""
    __tablename__ = "user_asked_questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), nullable=False)
    question_id = Column(Integer, nullable=False)
    timestamp = Column(TIMESTAMP, server_default=func.now())

class TeacherSuggestion(Base):
    """Model for teachers to send suggestions/feedback to students"""
    __tablename__ = "teacher_suggestions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    teacher_username = Column(String(255), nullable=False)
    student_username = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    is_read = Column(Integer, default=0) # 0 for unread, 1 for read
    timestamp = Column(TIMESTAMP, server_default=func.now())

class OTPVerification(Base):
    """Temporary storage for email verification OTPs"""
    __tablename__ = "otp_verifications"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), nullable=False, unique=True)
    otp_code = Column(String(6), nullable=False)
    username = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    branch = Column(String(50))
    role = Column(String(20), default='student')
    expires_at = Column(TIMESTAMP, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())