#!/bin/bash

echo "🚀 Setting up Legal Discovery Analysis System for M4 MacBook Pro"
echo "Python version: $(python3 --version)"
echo "==============================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

# Check if we're on Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    print_status "Detected Apple Silicon M4 - using optimized setup"
else
    print_warning "Not detected as Apple Silicon - continuing anyway"
fi

# Install Homebrew dependencies first (critical for M4)
install_homebrew_deps() {
    echo "Installing Homebrew dependencies..."
    
    if ! command -v brew &> /dev/null; then
        print_error "Homebrew not found. Please install it first:"
        echo "Visit: https://brew.sh/"
        exit 1
    fi
    
    # Core multimedia tools
    print_status "Installing FFmpeg..."
    brew install ffmpeg
    
    print_status "Installing Tesseract..."
    brew install tesseract
    
    print_status "Installing audio dependencies..."
    brew install portaudio
    brew install libsndfile
    
    # Python build dependencies (important for M4)
    print_status "Installing Python build tools..."
    brew install cmake
    brew install pkg-config
    
    print_status "Homebrew dependencies installed"
}

# Set up Python environment for M4
setup_python_env() {
    echo "Setting up Python environment for M4..."
    
    # Upgrade pip and build tools first
    python3 -m pip install --upgrade pip setuptools wheel cython
    
    # Set environment variables for M4 compatibility
    export ARCHFLAGS="-arch arm64"
    export _PYTHON_HOST_PLATFORM="macosx-11.0-arm64"
    export MACOSX_DEPLOYMENT_TARGET="11.0"
    
    print_status "Python environment configured for M4"
}

# Install core dependencies in stages (prevents conflicts on M4)
install_core_deps() {
    echo "Installing core dependencies in stages..."
    
    # Stage 1: Core framework
    print_status "Stage 1: Core framework..."
    pip3 install fastapi uvicorn[standard] python-multipart jinja2 python-dotenv
    pip3 install pydantic pydantic-settings sqlalchemy aiofiles websockets requests
    
    # Stage 2: Data processing (order matters on M4)
    print_status "Stage 2: Data processing..."
    pip3 install numpy pandas
    pip3 install pillow
    pip3 install opencv-python
    
    # Stage 3: AI and ML libraries
    print_status "Stage 3: AI and ML..."
    pip3 install sentence-transformers
    pip3 install chromadb
    pip3 install openai
    
    # Stage 4: Document processing
    print_status "Stage 4: Document processing..."
    pip3 install pypdf2 pdfplumber pymupdf
    pip3 install python-docx beautifulsoup4 openpyxl
    pip3 install pytesseract
    
    # Stage 5: Audio processing (can be tricky on M4)
    print_status "Stage 5: Audio processing..."
    pip3 install pydub soundfile mutagen
    pip3 install librosa  # This might take a while on M4
    
    # Stage 6: Whisper (special handling for M4)
    print_status "Stage 6: Installing Whisper..."
    if pip3 install openai-whisper; then
        print_status "Whisper installed successfully"
    else
        print_warning "Whisper installation failed - trying alternative approach"
        pip3 install --no-deps openai-whisper
        pip3 install torch torchvision torchaudio
    fi
    
    # Stage 7: Video processing (skip MoviePy, use FFmpeg)
    print_status "Stage 7: Video processing..."
    pip3 install ffmpeg-python
    print_warning "Skipping MoviePy (often problematic on M4) - using FFmpeg directly"
    
    # Stage 8: Optional OCR (EasyOCR can be problematic on M4)
    print_status "Stage 8: Attempting EasyOCR installation..."
    if pip3 install easyocr; then
        print_status "EasyOCR installed successfully"
    else
        print_warning "EasyOCR installation failed - will use Tesseract only"
    fi
    
    # Stage 9: Background processing
    print_status "Stage 9: Background processing..."
    pip3 install celery[redis] redis
    
    # Stage 10: Additional tools
    print_status "Stage 10: Additional tools..."
    pip3 install python-magic matplotlib scikit-learn scikit-image striprtf
    
    print_status "All core dependencies installed"
}

# Download models with M4 optimizations
download_models() {
    echo "Downloading required models..."
    
    # Download spaCy model
    python3 -m spacy download en_core_web_sm
    
    # Download NLTK data
    python3 -c "
import nltk
try:
    nltk.download('punkt')
    nltk.download('stopwords')  
    nltk.download('wordnet')
    print('✅ NLTK data downloaded')
except Exception as e:
    print(f'⚠️  NLTK download issue: {e}')
"
    
    # Test Whisper model download
    print_status "Testing Whisper model..."
    python3 -c "
import whisper
try:
    model = whisper.load_model('base')
    print('✅ Whisper base model loaded successfully')
except Exception as e:
    print(f'⚠️  Whisper model issue: {e}')
"
    
    print_status "Models downloaded"
}

# Create M4-optimized configuration
create_m4_config() {
    echo "Creating M4-optimized configuration..."
    
    mkdir -p data/logs data/vector_db data/temp_processing discovery_sets analysis_outputs
    
    if [ ! -f .env ]; then
        cat > .env << 'EOF'
# M4 MacBook Pro Optimized Configuration

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database
DATABASE_URL=sqlite:///./legal_discovery.db

# Vector Store
CHROMA_PERSIST_DIRECTORY=./data/vector_db

# File Storage
UPLOAD_DIRECTORY=./discovery_sets
TEMP_DIRECTORY=./data/temp_processing
MAX_FILE_SIZE=500000000

# OCR Configuration (M4 optimized paths)
TESSERACT_PATH=/opt/homebrew/bin/tesseract
OCR_LANGUAGE=eng
OCR_CONFIDENCE_THRESHOLD=60.0

# Audio Processing (M4 optimized)
WHISPER_MODEL=base
WHISPER_DEVICE=cpu  # Use CPU on M4 for stability

# Video Processing (M4 optimized)
MAX_VIDEO_DURATION=7200
VIDEO_FRAME_INTERVAL=30

# Background Processing
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0

# Logging
LOG_LEVEL=INFO
LOG_FILE=./data/logs/analysis.log

# M4 Specific Settings
ARCHFLAGS=-arch arm64
MACOSX_DEPLOYMENT_TARGET=11.0
EOF
        print_status "M4-optimized configuration created"
    fi
}

# Test installation on M4
test_m4_installation() {
    echo "Testing installation on M4..."
    
    python3 << 'EOF'
import sys
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")

def test_import(module_name, description):
    try:
        __import__(module_name)
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description}: {e}")
        return False

success = 0
total = 0

# Test core components
modules = [
    ("fastapi", "FastAPI"),
    ("whisper", "OpenAI Whisper"),
    ("cv2", "OpenCV"),
    ("librosa", "Librosa"),
    ("chromadb", "ChromaDB"),
    ("PIL", "Pillow"),
    ("pytesseract", "Tesseract"),
    ("pydub", "PyDub"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy")
]

for module, desc in modules:
    total += 1
    if test_import(module, desc):
        success += 1

# Test optional components
try:
    import easyocr
    print("✅ EasyOCR (optional)")
    success += 1
except ImportError:
    print("⚠️  EasyOCR not available (will use Tesseract only)")
total += 1

# Test FFmpeg
try:
    import subprocess
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
    if result.returncode == 0:
        print("✅ FFmpeg")
        success += 1
    else:
        print("❌ FFmpeg")
except:
    print("❌ FFmpeg")
total += 1

# Test Tesseract
try:
    import pytesseract
    from PIL import Image
    test_img = Image.new('RGB', (100, 50), color='white')
    pytesseract.image_to_string(test_img)
    print("✅ Tesseract OCR")
    success += 1
except Exception as e:
    print(f"❌ Tesseract OCR: {e}")
total += 1

print(f"\n🎉 M4 Installation Summary: {success}/{total} components working")

if success >= total - 2:
    print("✅ Installation successful for M4! Ready to start.")
    print("\nOptimization notes for M4:")
    print("- Using CPU-based processing (more stable)")
    print("- FFmpeg used instead of MoviePy")
    print("- Tesseract as primary OCR engine")
else:
    print("⚠️  Some components failed. Check errors above.")
EOF
}

# Main installation process
main() {
    install_homebrew_deps
    setup_python_env
    install_core_deps
    download_models
    create_m4_config
    test_m4_installation
    
    echo ""
    echo "==============================================================="
    echo "🎉 M4 MacBook Pro Setup Complete!"
    echo ""
    echo "Next steps:"
    echo "1. Update your OpenAI API key in .env file"
    echo "2. Start Redis: brew services start redis"
    echo "3. Start the application:"
    echo "   python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    echo "4. Open http://localhost:8000"
    echo ""
    echo "M4-specific optimizations applied:"
    echo "- CPU-based processing for stability"
    echo "- FFmpeg for video processing"
    echo "- Optimized dependency installation order"
    echo "- Apple Silicon build flags set"
}

# Run main function
main