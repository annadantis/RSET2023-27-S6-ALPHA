from fastapi import APIRouter, HTTPException
import requests
import ollama
from typing import List, Dict, Optional
from pydantic import BaseModel
import time
from concurrent.futures import ThreadPoolExecutor
from newspaper import Article
import pyttsx3
import threading
import subprocess
import sys
import queue

router = APIRouter(prefix="/news", tags=["news"])

# Robust TTS Process Management for Windows
class SpeechManager:
    def __init__(self):
        self._current_process = None
        self._lock = threading.Lock()

    def speak(self, text):
        """Terminate current speech and start new process-based TTS."""
        with self._lock:
            self.stop_locked()
            
            # Use sys.executable to ensure we use the same python environment
            # Small script to handle the TTS in isolation
            script = f"import pyttsx3; e=pyttsx3.init(); e.setProperty('rate', 175); e.say({repr(text)}); e.runAndWait()"
            
            try:
                print(f"🎙️ SpeechManager: Launching TTS process...")
                self._current_process = subprocess.Popen(
                    [sys.executable, "-c", script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception as e:
                print(f"⚠️ SpeechManager Launch Error: {e}")

    def stop(self):
        """Public stop method."""
        with self._lock:
            self.stop_locked()

    def stop_locked(self):
        """Internal stop to be called under lock."""
        if self._current_process and self._current_process.poll() is None:
            print(f"🛑 SpeechManager: Terminating current speech...")
            try:
                # Try graceful termination first
                self._current_process.terminate()
                # Wait briefly, then kill if still alive
                try:
                    self._current_process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self._current_process.kill()
            except Exception as e:
                print(f"⚠️ Error stopping speech process: {e}")
        self._current_process = None

# Global manager instance
speech_manager = SpeechManager()

def speak(text):
    """Bridge for existing calls to speak()."""
    speech_manager.speak(text)

@router.post("/stop-speech")
def stop_news_speech():
    """Stop any ongoing news summary speech."""
    speech_manager.stop()
    return {"status": "success"}

def extract_article(url):
    """Extract and truncate article text to 1200 characters."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text[:1200]
    except Exception as e:
        print(f"⚠️ Article extraction failed for {url}: {e}")
        return ""


# Simple in-memory cache
news_cache = {
    "data": [],
    "last_updated": 0
}
CACHE_DURATION = 600  # 10 minutes

def fetch_story(story_id):
    """Worker function to fetch a single story's details (Synchronous)."""
    try:
        item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        res = requests.get(item_url, timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"Error fetching story {story_id}: {e}")
    return None

@router.get("/latest")
def get_latest_news():
    """
    Fetch top stories from Hacker News and filter for industry/tech trends.
    Using 'def' instead of 'async def' to run this blocking code in a thread pool.
    """
    global news_cache
    
    current_time = time.time()
    if news_cache["data"] and (current_time - news_cache["last_updated"] < CACHE_DURATION):
        return news_cache["data"]

    keywords = [
        "tech", "software", "developer", "hiring", "job", "career", "AI", "LLM", 
        "engineering", "coding", "startup", "recruitment", "salary", "interview", 
        "placement", "algorithm", "system design", "cloud", "aws", "google", 
        "microsoft", "apple", "nvidia", "meta", "web", "mobile", "frontend", "backend"
    ]

    try:
        print("📰 Fetching latest industry news...")
        # 1. Get top story IDs
        top_ids_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        response = requests.get(top_ids_url, timeout=10)
        response.raise_for_status()
        top_ids = response.json()[:80]

        # 2. Parallel fetch story details
        with ThreadPoolExecutor(max_workers=20) as executor:
            raw_items = list(executor.map(fetch_story, top_ids))

        stories = []
        for item in raw_items:
            if not item: continue
            
            title = item.get("title", "").lower()
            if any(kw in title for kw in keywords):
                if item.get("type") == "story" and "url" in item:
                    stories.append({
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "score": item.get("score"),
                        "time": item.get("time"),
                        "by": item.get("by")
                    })
            
            if len(stories) >= 15: break

        # Fallback to top stories if not enough industry ones found
        if len(stories) < 5:
            for item in raw_items:
                if not item: continue
                if any(s["id"] == item.get("id") for s in stories): continue
                if item.get("type") == "story" and "url" in item:
                    stories.append({
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "score": item.get("score"),
                        "time": item.get("time"),
                        "by": item.get("by")
                    })
                if len(stories) >= 10: break

        print(f"✅ Successfully fetched {len(stories)} stories.")
        news_cache["data"] = stories
        news_cache["last_updated"] = current_time
        return stories

    except Exception as e:
        print(f"❌ Error fetching news: {e}")
        if news_cache["data"]: return news_cache["data"]
        raise HTTPException(status_code=500, detail=f"Failed to fetch industry news: {str(e)}")

class SummaryRequest(BaseModel):
    title: str
    url: Optional[str] = None

@router.post("/summary")
def get_news_summary(request: SummaryRequest):
    """
    Generate a 2-sentence summary using the fast pipeline (Extract -> phi3:mini -> Speak).
    """
    try:
        print(f"🤖 Processing fast pipeline for: {request.title}")
        
        # 1. Extraction (if URL available)
        article_text = ""
        if request.url:
            article_text = extract_article(request.url)
        
        # Fallback to title if extraction fails or no URL
        text_to_summarize = article_text if article_text and len(article_text) > 100 else request.title
        
        # 2. Fast Summarization with phi3:mini
        prompt = f"Summarize this tech news in 2 short sentences:\n{text_to_summarize}"
        
        response = ollama.generate(
            model='phi3:mini',
            prompt=prompt,
            options={'num_predict': 150, 'temperature': 0.7}
        )
        
        summary = response.get('response', '').strip()
        if not summary:
            summary = f"Summary of {request.title}: {text_to_summarize[:100]}..."
            
        # 3. Instant TTS
        speak(summary)
            
        return {"summary": summary}
    except Exception as e:
        print(f"⚠️ Error in fast news pipeline: {e}")
        return {"summary": request.title}

# Cache for the collective briefing
briefing_cache = {
    "text": "",
    "last_updated": 0
}

@router.get("/trends-briefing")
def get_trends_briefing():
    """
    Generates a collective AI briefing from the current top news stories.
    Using 'def' allows this blocking AI call to run in a thread pool.
    """
    global briefing_cache
    current_time = time.time()
    
    # Return cached briefing if it's less than 30 minutes old
    if briefing_cache["text"] and (current_time - briefing_cache["last_updated"] < 1800):
        return {"briefing": briefing_cache["text"]}

    try:
        # 1. Get the latest news from our existing function (which uses its own cache)
        latest_news = get_latest_news()
        
        if not latest_news:
            return {"briefing": "Currently, there are no major tech trends to report. Please check back later."}

        # 2. Extract titles (limit to top 10 for context size)
        titles = [item.get("title", "") for item in latest_news[:10] if item.get("title")]
        titles_text = "\n- ".join(titles)

        print("🤖 Generating collective industry briefing...")
        prompt = f"""
        You are an expert career advisor for engineering students.
        Based on the following recent Hacker News tech headlines, write a cohesive, professional 3-sentence briefing summarizing the current state or major movements in the tech industry. 
        Focus on what these trends mean for a student preparing for placements (e.g., AI adoption, hiring shifts, new technologies).
        Do not list the headlines, synthesize them into a paragraph. Do not use quotes or introductory phrases. Make it sound like a professional radio briefing.

        Headlines:
        - {titles_text}
        """
        
        response = ollama.generate(
            model='phi3:mini',
            prompt=prompt,
            options={'num_predict': 200, 'temperature': 0.7}
        )
        
        briefing = response.get('response', '').strip()
        
        if not briefing:
            return {"briefing": "The industry is currently focused on rapid technological advancements. Continue focusing on core engineering skills and AI adaptability."}
            
        # Update cache
        briefing_cache["text"] = briefing
        briefing_cache["last_updated"] = current_time
            
        return {"briefing": briefing}
        
    except Exception as e:
        print(f"⚠️ Error generating trends briefing: {e}")
        return {"briefing": "We are currently unable to generate the industry briefing. Please stay focused on your core technical preparation."}
