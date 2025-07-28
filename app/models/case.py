from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

Base = declarative_base()

class Case(Base):
    __tablename__ = "cases"
    
    id = Column(String, primary_key=True, index=True)
    case_number = Column(String, unique=True, index=True)
    case_name = Column(String)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CaseCreate(BaseModel):
    case_number: str
    case_name: str
    description: Optional[str] = None

class CaseResponse(BaseModel):
    id: str
    case_number: str
    case_name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True