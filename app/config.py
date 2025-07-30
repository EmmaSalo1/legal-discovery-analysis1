import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///./legal_discovery.db"
    
    # AI Services
    openai_api_key: str = "your_openai_api_key_here"
    openai_model: str = "gpt-4-turbo-preview"
    
    # Whisper Configuration
    whisper_model: str = "base"  # tiny, base, small, medium, large
    whisper_device: str = "cpu"  # cpu or mps (for Mac M1/M2)
    
    # ChromaDB
    chroma_persist_directory: str = "./data/vector_db"
    
    # File Storage
    upload_directory: str = "./discovery_sets"
    temp_directory: str = "./data/temp_processing"
    max_file_size: int = 500000000  # 500MB
    
    # Multimedia Processing
    supported_audio_formats: List[str] = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]
    supported_video_formats: List[str] = [".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv"]
    supported_image_formats: List[str] = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]  # Removed .pdf
    
    # OCR Settings - Try multiple possible paths
    tesseract_path: str = None  # Will be auto-detected
    ocr_language: str = "eng"
    ocr_confidence_threshold: float = 60.0
    
    # Audio Processing
    audio_chunk_length: int = 30  # seconds
    silence_threshold: float = -40.0  # dB
    
    # Video Processing
    video_frame_interval: int = 30  # Extract frame every N seconds
    max_video_duration: int = 7200  # 2 hours max
    
    # Background Processing
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./data/logs/analysis.log"
    multimedia_log_file: str = "./data/logs/multimedia_processing.log"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-detect Tesseract path if not set
        if not self.tesseract_path:
            self.tesseract_path = self._find_tesseract()
    
    def _find_tesseract(self) -> str:
        """Auto-detect Tesseract installation path"""
        possible_paths = [
            "/opt/homebrew/bin/tesseract",  # Mac M1/M2 Homebrew
            "/usr/local/bin/tesseract",     # Mac Intel Homebrew
            "/usr/bin/tesseract",           # Linux
            "tesseract",                    # Windows (in PATH)
            "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",  # Windows default
        ]
        
        for path in possible_paths:
            if os.path.exists(path) or path == "tesseract":
                return path
        
        # If not found, return None and disable OCR
        return None
    
    class Config:
        env_file = ".env"

settings = Settings()