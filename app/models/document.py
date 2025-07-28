from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, index=True)
    file_path = Column(String)
    file_name = Column(String)
    file_type = Column(String)
    file_size = Column(Integer)
    content = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

class DocumentCreate(BaseModel):
    case_id: str
    file_path: str
    file_name: str
    file_type: str
    file_size: int
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DocumentResponse(BaseModel):
    id: int
    case_id: str
    file_path: str
    file_name: str
    file_type: str
    file_size: int
    created_at: datetime
    processed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True