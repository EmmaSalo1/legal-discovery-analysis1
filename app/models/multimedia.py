from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

Base = declarative_base()

class MultimediaDocument(Base):
    __tablename__ = "multimedia_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, index=True)
    file_path = Column(String)
    file_type = Column(String)  # audio, video, image
    file_format = Column(String)  # mp3, mp4, jpg, etc.
    file_size = Column(Integer)
    duration = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    
    # Processing status
    processing_status = Column(String, default="pending")
    
    # Extracted content
    transcript_text = Column(Text, nullable=True)
    ocr_text = Column(Text, nullable=True)
    
    # Metadata and analysis
    metadata = Column(JSON)
    timestamp_created = Column(DateTime, default=datetime.utcnow)

class MultimediaDocumentCreate(BaseModel):
    case_id: str
    file_path: str
    file_type: str
    file_format: str
    file_size: int
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None