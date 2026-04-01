import os
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool, QueuePool
from urllib.parse import quote_plus
import pymysql

# Install pymysql as MySQLdb
pymysql.install_as_MySQLdb()

Base = declarative_base()

# --- MYSQL CONFIGURATION ---
# --- MYSQL CONFIGURATION ---
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "Anna4@aa")
DB_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
DB_PORT = os.getenv("MYSQL_PORT", "3306")
DB_NAME = os.getenv("MYSQL_DATABASE", "placement_app")

# Create MySQL connection URL
MYSQL_URL = f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

print(f"Connecting to MySQL: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Create engine with improved connection handling
mysql_engine = create_engine(
    MYSQL_URL,
    poolclass=QueuePool,
    pool_size=5,              # Number of connections to maintain
    max_overflow=10,          # Maximum overflow connections
    pool_pre_ping=True,       # Test connections before use
    pool_recycle=3600,        # Recycle connections after 1 hour
    pool_timeout=30,          # Timeout for getting connection from pool
    echo=False,               # Set to True for SQL debugging
    connect_args={
        "connect_timeout": 10,
        "read_timeout": 30,
        "write_timeout": 30
    }
)

# Add event listeners to handle disconnections
@event.listens_for(mysql_engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Called when a new DB connection is created"""
    # Set connection parameters
    with dbapi_conn.cursor() as cursor:
        cursor.execute("SET SESSION wait_timeout=28800")  # 8 hours
        cursor.execute("SET SESSION interactive_timeout=28800")

@event.listens_for(mysql_engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Called when connection is retrieved from pool"""
    # Ping to check if connection is alive
    try:
        dbapi_conn.ping(reconnect=True)
    except:
        # If ping fails, invalidate the connection
        connection_record.invalidate()
        raise

# Create session factory
SessionLocal = sessionmaker(
    bind=mysql_engine, 
    autoflush=False, 
    autocommit=False,
    expire_on_commit=False
)


def get_db():
    """
    Dependency for FastAPI routes with automatic cleanup.
    Handles connection errors gracefully.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        print(f"Database error in request: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    try:
        from models import (
            User, Score, Question, GDEvaluation, WeeklyStat, 
            GDTopic, GDResult, InterviewDetail, QuizAnswer, 
            InterviewQuestion, UserAskedQuestion, TeacherSuggestion,
            OTPVerification
        )
        Base.metadata.create_all(bind=mysql_engine)
        
        # Ensure teacher_username column exists (handle existing tables from older versions)
        # Standard MySQL doesn't support "ADD COLUMN IF NOT EXISTS"
        try:
            with mysql_engine.connect() as conn:
                # 1. Check if column exists
                check_sql = text("""
                    SELECT COUNT(*) FROM information_schema.columns 
                    WHERE table_schema = :db_name 
                    AND table_name = 'teacher_suggestions' 
                    AND column_name = 'teacher_username'
                """)
                column_exists = conn.execute(check_sql, {"db_name": DB_NAME}).scalar() > 0
                
                if not column_exists:
                    print(f"🔧 Migrating: Adding 'teacher_username' column to 'teacher_suggestions' table...")
                    conn.execute(text("ALTER TABLE teacher_suggestions ADD COLUMN teacher_username VARCHAR(255) AFTER id"))
                    conn.commit()
                    print("✅ Migration successful!")
                else:
                    print("✅ Column 'teacher_username' already exists.")
        except Exception as e:
            print(f"Migration Note: {e}")

        print("Database tables created/verified successfully")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        raise


def test_connection():
    """Test MySQL connection with retry logic"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            connection = mysql_engine.connect()
            result = connection.execute(text("SELECT 1"))
            connection.close()
            print("MySQL connection successful")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  Connection attempt {attempt + 1} failed, retrying...")
                import time
                time.sleep(retry_delay)
            else:
                print(f"MySQL connection failed after {max_retries} attempts: {e}")
                print("\nTroubleshooting:")
                print("1. Make sure XAMPP MySQL is running")
                print("2. Check MySQL didn't crash - restart it in XAMPP")
                print(f"3. Verify database '{DB_NAME}' exists")
                print("4. Check MySQL error logs in XAMPP")
                return False
    
    return False


def create_database_if_not_exists():
    """Create the database if it doesn't exist"""
    try:
        temp_url = f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/"
        temp_engine = create_engine(temp_url, poolclass=NullPool)
        
        connection = temp_engine.connect()
        connection.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
        connection.close()
        temp_engine.dispose()
        
        print(f"Database '{DB_NAME}' ready")
        return True
    except Exception as e:
        print(f"Error creating database: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DATABASE SETUP TEST")
    print("="*60 + "\n")
    
    if test_connection():
        print("\nDatabase connection is working!")
        
        try:
            init_db()
            print("\nAll tests passed!")
        except Exception as e:
            print(f"\n⚠️  Warning: Could not create tables: {e}")
            print("You may need to run setup_mysql.sql first")
    else:
        print("\nDatabase connection failed!")
        print("\nPlease check:")
        print("1. XAMPP MySQL service is running (and stays running)")
        print("2. Database credentials are correct")
        print(f"3. Database '{DB_NAME}' exists")
        print("4. Check XAMPP MySQL logs for errors")