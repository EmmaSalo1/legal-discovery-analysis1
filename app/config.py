# Update your app/config.py for M4 MacBook Pro optimization

import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///./legal_discovery.db"
    
    # AI Services
    openai_api_key: str = "your_openai_api_key_here"
    openai_model: str = "gpt-4-turbo-preview"
    
    # Whisper Configuration (M4 optimized)
    whisper_model: str = "base"  # Use base model for better M4 performance
    whisper_device: str = "cpu"  # CPU more stable on M4 than MPS
    
    # ChromaDB
    chroma_persist_directory: str = "./data/vector_db"
    
    # File Storage
    upload_directory: str = "./discovery_sets"
    temp_directory: str = "./data/temp_processing"
    max_file_size: int = 500000000  # 500MB
    
    # Multimedia Processing (M4 optimized)
    supported_audio_formats: List[str] = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]
    supported_video_formats: List[str] = [".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv"]
    supported_image_formats: List[str] = [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".pdf"]
    
    # OCR Settings
    # Default to the typical Linux path; override in .env for macOS/Homebrew.
    tesseract_path: str = "/usr/bin/tesseract"
    ocr_language: str = "eng"
    ocr_confidence_threshold: float = 60.0
    
    # Audio Processing (M4 optimized)
    audio_chunk_length: int = 30  # seconds
    silence_threshold: float = -40.0  # dB
    
    # Video Processing (M4 optimized - prefer FFmpeg)
    video_frame_interval: int = 30  # Extract frame every N seconds
    max_video_duration: int = 7200  # 2 hours max
    prefer_ffmpeg: bool = True  # Use FFmpeg over MoviePy on M4
    
    # Background Processing
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./data/logs/analysis.log"
    multimedia_log_file: str = "./data/logs/multimedia_processing.log"
    
    # M4 Specific Settings
    apple_silicon_optimized: bool = True
    use_mps_if_available: bool = False  # Disable MPS for stability
    cpu_processing_preferred: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()

# M4-specific helpers
def is_apple_silicon():
    """Check if running on Apple Silicon"""
    import platform
    return platform.machine() == 'arm64' and platform.system() == 'Darwin'

def get_optimal_processing_device():
    """Get optimal processing device for M4"""
    if is_apple_silicon():
        return "cpu" 
    return "cpu"  # Default to CPU for compatibility