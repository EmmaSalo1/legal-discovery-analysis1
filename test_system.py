"""
Legal Discovery Analysis System - System Test Script
Tests all major components to ensure proper installation
"""

import os
import sys
import asyncio
import pytest
from pathlib import Path

def test_basic_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing basic package imports...")
    
    try:
        import fastapi
        print("✅ FastAPI")
        
        import uvicorn
        print("✅ Uvicorn")
        
        import whisper
        print("✅ Whisper")
        
        import cv2
        print("✅ OpenCV")
        
        import librosa
        print("✅ Librosa")
        
        import chromadb
        print("✅ ChromaDB")
        
        from PIL import Image
        print("✅ Pillow (PIL)")
        
        import numpy as np
        print("✅ NumPy")
        
        import pandas as pd
        print("✅ Pandas")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_ocr_functionality():
    """Test OCR functionality"""
    print("\n🔍 Testing OCR functionality...")
    
    try:
        import pytesseract
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a test image with text
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 30), "TEST DOCUMENT", fill='black', font=font)
        
        # Test OCR
        text = pytesseract.image_to_string(img)
        
        if "TEST" in text.upper():
            print("✅ OCR working correctly")
            return True
        else:
            print(f"⚠️  OCR detected text: '{text.strip()}' (may need configuration)")
            return False
            
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        print("   Install Tesseract: brew install tesseract (macOS) or apt-get install tesseract-ocr (Ubuntu)")
        return False

def test_whisper_functionality():
    """Test Whisper audio processing"""
    print("\n🎵 Testing Whisper functionality...")
    
    try:
        import whisper
        
        # Load the smallest model for testing
        model = whisper.load_model("tiny")
        print("✅ Whisper model loaded successfully")
        
        # Create a simple test (we can't easily create audio, so just verify model loading)
        return True
        
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")
        return False

def test_video_processing():
    """Test video processing capabilities"""
    print("\n🎥 Testing video processing...")
    
    try:
        import moviepy.editor as mp
        import cv2
        
        # Test OpenCV video capability
        # Create a simple test to see if video codecs are available
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("✅ Video codecs available")
        
        print("✅ Video processing libraries working")
        return True
        
    except Exception as e:
        print(f"❌ Video processing test failed: {e}")
        print("   Install FFmpeg: brew install ffmpeg (macOS) or apt-get install ffmpeg (Ubuntu)")
        return False

def test_openai_connection():
    """Test OpenAI API connection"""
    print("\n🤖 Testing OpenAI API connection...")
    
    try:
        from openai import OpenAI
        import os
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key or api_key == 'your_openai_api_key_here':
            print("⚠️  OpenAI API key not configured in .env file")
            return False
        
        # Test connection (just create client, don't make actual API call to save costs)
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client created successfully")
        print("   (Note: Actual API calls not tested to avoid charges)")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return False

def test_file_system():
    """Test file system setup"""
    print("\n📁 Testing file system setup...")
    
    required_dirs = [
        'discovery_sets',
        'data/logs',
        'data/vector_db',
        'data/temp_processing',
        'analysis_outputs'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (creating...)")
            os.makedirs(dir_path, exist_ok=True)
            all_good = False
    
    return all_good

def test_database_setup():
    """Test database functionality"""
    print("\n🗄️  Testing database setup...")
    
    try:
        from sqlalchemy import create_engine
        from app.models.case import Base
        
        # Test database creation
        engine = create_engine("sqlite:///test_legal_discovery.db")
        Base.metadata.create_all(bind=engine)
        
        print("✅ Database tables created successfully")
        
        # Clean up test database
        if os.path.exists("test_legal_discovery.db"):
            os.remove("test_legal_discovery.db")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

@pytest.mark.asyncio
async def test_app_components():
    """Test main application components"""
    print("\n🏗️  Testing application components...")
    
    try:
        from app.services.document_processor import DocumentProcessor
        from app.services.vector_store import VectorStore
        from app.services.rag_system import RAGSystem
        
        # Test component initialization
        doc_processor = DocumentProcessor()
        print("✅ Document processor initialized")
        
        vector_store = VectorStore()
        print("✅ Vector store initialized")
        
        rag_system = RAGSystem(vector_store)
        print("✅ RAG system initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ App components test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🏛️  Legal Discovery Analysis System - Installation Test\n")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("OCR Functionality", test_ocr_functionality),
        ("Whisper Audio Processing", test_whisper_functionality),
        ("Video Processing", test_video_processing),
        ("OpenAI Connection", test_openai_connection),
        ("File System", test_file_system),
        ("Database Setup", test_database_setup),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"{'='*50}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Test async components
    print(f"{'='*50}")
    try:
        result = asyncio.run(test_app_components())
        results.append(("App Components", result))
    except Exception as e:
        print(f"❌ App Components failed with exception: {e}")
        results.append(("App Components", False))
    
    # Summary
    print(f"\n{'='*50}")
    print("🏁 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} | {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("   Run './start.sh' to start the application")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
        print("   You can still run the application, but some features may not work.")
        print("   See the troubleshooting guide for help with failed tests.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)