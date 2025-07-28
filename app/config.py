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
    
    # Multimedia Processing - FIXED: Removed PDF from image formats
    supported_audio_formats: List[str] = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]
    supported_video_formats: List[str] = [".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv"]
    supported_image_formats: List[str] = [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif"]  # Removed .pdf
    
    # OCR Settings
    tesseract_path: str = "/opt/homebrew/bin/tesseract"  # Mac Homebrew path
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
    
    class Config:
        env_file = ".env"

settings = Settings()