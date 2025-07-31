from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.requests import Request
from starlette.middleware.cors import CORSMiddleware
import logging
import os
import mimetypes
from typing import List, Dict, Any, Optional
import json
import asyncio
import aiofiles
from datetime import datetime
import uuid

from app.config import settings
from app.models.case import Case, CaseCreate
from app.services.document_processor import EnhancedDocumentProcessor
from app.services.rag_system import EnhancedRAGSystem
from app.services.vector_store import EnhancedVectorStore
from app.utils.database import get_db, engine, Base

# Create tables
Base.metadata.create_all(bind=engine)

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legal Discovery Analysis System",
    version="3.0.0",
    description="Advanced legal discovery system with comprehensive multimedia support"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create required directories
os.makedirs(settings.upload_directory, exist_ok=True)
os.makedirs(settings.temp_directory, exist_ok=True)
os.makedirs("./data/logs", exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# Initialize enhanced services
document_processor = EnhancedDocumentProcessor()
vector_store = EnhancedVectorStore()
rag_system = EnhancedRAGSystem(vector_store)

# Enhanced WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.case_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, case_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        if case_id not in self.case_connections:
            self.case_connections[case_id] = []
        self.case_connections[case_id].append(websocket)
        logger.info(f"WebSocket connected for case {case_id}")

    def disconnect(self, websocket: WebSocket, case_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if case_id in self.case_connections and websocket in self.case_connections[case_id]:
            self.case_connections[case_id].remove(websocket)
        logger.info(f"WebSocket disconnected for case {case_id}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def broadcast_to_case(self, message: dict, case_id: str):
        """Broadcast message to all connections for a specific case"""
        if case_id in self.case_connections:
            disconnected = []
            for websocket in self.case_connections[case_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to case {case_id}: {e}")
                    disconnected.append(websocket)
            
            # Remove disconnected websockets
            for ws in disconnected:
                self.disconnect(ws, case_id)

    async def broadcast_to_case(self, message: dict, case_id: str):
        """Send a message to all active WebSocket connections for a case."""
        connections = self.case_connections.get(case_id, [])
        for connection in list(connections):
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"WebSocket broadcast error: {e}")
                self.disconnect(connection, case_id)

manager = ConnectionManager()

# Enhanced utility functions
def get_file_type(file_path: str) -> str:
    """Determine file type with enhanced multimedia detection"""
    mime_type, _ = mimetypes.guess_type(file_path)
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension in settings.supported_audio_formats:
        return 'audio'
    elif extension in settings.supported_video_formats:
        return 'video'
    elif extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
        return 'image'
    elif extension in ['.pdf', '.docx', '.doc', '.txt', '.rtf']:
        return 'document'
    else:
        # Fallback to mime type detection
        if mime_type:
            if mime_type.startswith('audio/'):
                return 'audio'
            elif mime_type.startswith('video/'):
                return 'video'
            elif mime_type.startswith('image/'):
                return 'image'
            elif 'document' in mime_type or 'text' in mime_type:
                return 'document'
        
        return 'document'  # Default fallback

async def process_file_background(file_path: str, case_id: str, file_type: str, processing_id: str):
    """Enhanced background processing for multimedia files"""
    try:
        logger.info(f"Starting background processing: {file_path} (type: {file_type})")
        
        # Send processing start notification
        await manager.broadcast_to_case({
            'type': 'processing_started',
            'processing_id': processing_id,
            'filename': os.path.basename(file_path),
            'file_type': file_type,
            'status': 'processing'
        }, case_id)
        
        # Process the file with enhanced processor
        result = await document_processor.process_document(file_path, case_id)
        
        # Add to vector store if successful
        if result.get('processing_status') == 'completed':
            doc_id = await vector_store.add_document(result, case_id)
            result['vector_store_id'] = doc_id
            logger.info(f"Added to vector store with ID: {doc_id}")
        
        # Notify completion via WebSocket
        await manager.broadcast_to_case({
            'type': 'processing_complete',
            'processing_id': processing_id,
            'filename': os.path.basename(file_path),
            'file_type': file_type,
            'status': 'completed' if result.get('processing_status') == 'completed' else 'failed',
            'result_summary': result.get('summary', 'Processing completed'),
            'has_content': bool(result.get('content') or 
                              result.get('transcript', {}).get('text') or 
                              result.get('ocr_results', {}).get('combined_text'))
        }, case_id)
        
        logger.info(f"Background processing completed successfully: {file_path}")
        
    except Exception as e:
        logger.error(f"Background processing failed for {file_path}: {str(e)}", exc_info=True)
        await manager.broadcast_to_case({
            'type': 'processing_complete',
            'processing_id': processing_id,
            'filename': os.path.basename(file_path),
            'file_type': file_type,
            'status': 'failed',
            'error': str(e)
        }, case_id)

# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Main dashboard"""
    return templates.TemplateResponse("case_dashboard.html", {"request": request})

@app.get("/case/{case_id}", response_class=HTMLResponse)
async def case_dashboard(request: Request, case_id: str):
    """Case dashboard"""
    return templates.TemplateResponse("case_dashboard.html", {
        "request": request,
        "case_id": case_id
    })

# API Routes
@app.post("/api/cases/")
async def create_case(case: CaseCreate):
    """Create a new case with enhanced directory structure"""
    try:
        # Create case directory
        case_dir = os.path.join(settings.upload_directory, case.case_number)
        os.makedirs(case_dir, exist_ok=True)
        
        # Create enhanced subdirectories for multimedia
        subdirs = [
            "plaintiff_production/documents",
            "plaintiff_production/audio", 
            "plaintiff_production/video",
            "plaintiff_production/images",
            "defendant_production/documents",
            "defendant_production/audio",
            "defendant_production/video", 
            "defendant_production/images",
            "third_party_production/documents",
            "third_party_production/multimedia",
            "court_filings",
            "analysis_outputs"
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(case_dir, subdir), exist_ok=True)
        
        logger.info(f"Created enhanced case structure: {case.case_number}")
        return {"id": case.case_number, **case.dict()}
        
    except Exception as e:
        logger.error(f"Error creating case: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cases/{case_id}/files")
async def list_case_files(case_id: str, file_type: Optional[str] = None):
    """List files with enhanced multimedia information"""
    try:
        case_dir = os.path.join(settings.upload_directory, case_id)
        if not os.path.exists(case_dir):
            raise HTTPException(status_code=404, detail="Case not found")
        
        files = []
        
        for root, dirs, filenames in os.walk(case_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, case_dir)
                
                detected_type = get_file_type(file_path)
                
                if file_type and detected_type != file_type:
                    continue
                
                # Enhanced file info with multimedia metadata
                file_info = {
                    'filename': filename,
                    'path': relative_path,
                    'type': detected_type,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                    'extension': os.path.splitext(filename)[1].lower()
                }
                
                # Add processing status check
                file_info['processing_status'] = await check_file_processing_status(file_path, case_id)
                
                files.append(file_info)
        
        # Sort by type and then by name
        files.sort(key=lambda x: (x['type'], x['filename']))
        
        # Calculate type statistics
        type_counts = {}
        for file in files:
            file_type = file['type']
            type_counts[file_type] = type_counts.get(file_type, 0) + 1
        
        return {
            'case_id': case_id, 
            'files': files, 
            'total_files': len(files),
            'type_counts': type_counts
        }
        
    except Exception as e:
        logger.error(f"Error listing case files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def check_file_processing_status(file_path: str, case_id: str) -> str:
    """Check if a file has been processed"""
    try:
        filename = os.path.basename(file_path)
        file_type = get_file_type(file_path)
        expected_id = f"{file_type}_{case_id}_{filename}"
        
        # Try to find in vector store
        results = await vector_store.search_documents(case_id, filename, limit=1)
        if results:
            return 'completed'
        else:
            return 'pending'
    except:
        return 'unknown'

@app.post("/api/cases/{case_id}/documents/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    case_id: str,
    files: List[UploadFile] = File(...),
    document_type: str = Form("general"),
    party: str = Form("plaintiff_production")
):
    """Enhanced upload with multimedia support"""
    try:
        uploaded_files = []
        
        for file in files:
            processing_id = str(uuid.uuid4())
            file_type = get_file_type(file.filename)
            
            # Determine subdirectory based on file type
            if file_type == 'audio':
                subdir = 'audio'
            elif file_type == 'video':
                subdir = 'video'
            elif file_type == 'image':
                subdir = 'images'
            else:
                subdir = 'documents'
            
            file_dir = os.path.join(settings.upload_directory, case_id, party, subdir)
            os.makedirs(file_dir, exist_ok=True)
            
            # Save file with size validation
            file_path = os.path.join(file_dir, file.filename)
            
            content = await file.read()
            
            # Check file size
            if len(content) > settings.max_file_size:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File {file.filename} exceeds maximum size limit"
                )
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            uploaded_files.append({
                'filename': file.filename,
                'file_path': file_path,
                'file_type': file_type,
                'processing_id': processing_id,
                'size': len(content),
                'party': party,
                'document_type': document_type
            })
            
            # Queue background processing
            background_tasks.add_task(
                process_file_background,
                file_path, case_id, file_type, processing_id
            )
            
            logger.info(f"Uploaded and queued for processing: {file.filename} ({file_type})")
        
        return {
            "message": f"Uploaded {len(uploaded_files)} files",
            "files": uploaded_files,
            "processing_note": "Files are being processed in the background. You'll receive notifications when complete."
        }
        
    except Exception as e:
        logger.error(f"Error uploading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cases/{case_id}/search")
async def search_case(case_id: str, query: str, file_types: Optional[str] = None, limit: int = 50):
    """Enhanced search with multimedia content support"""
    try:
        # Parse file types filter
        target_types = file_types.split(',') if file_types else ['audio', 'video', 'image', 'document']
        
        # Use enhanced RAG system for multimedia search
        results = await rag_system.search_multimedia_content(
            case_id=case_id,
            query=query,
            content_types=target_types,
            limit=limit
        )
        
        # Enhance results with multimedia-specific snippets
        enhanced_results = []
        for result in results:
            snippets = await extract_relevant_snippets(result, query)
            
            enhanced_result = {
                **result,
                'snippets': snippets,
                'relevance_score': result.get('similarity_score', 0),
                'content_summary': await generate_content_summary(result)
            }
            enhanced_results.append(enhanced_result)
        
        return {
            'case_id': case_id,
            'query': query,
            'results': enhanced_results,
            'total_results': len(enhanced_results),
            'search_types': target_types
        }
        
    except Exception as e:
        logger.error(f"Error searching case: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def extract_relevant_snippets(result: Dict, query: str) -> List[Dict]:
    """Extract relevant snippets from multimedia content"""
    snippets = []
    file_type = result.get('file_type', 'document')
    
    try:
        if file_type == 'audio' and 'transcript' in result:
            text = result['transcript'].get('text', '')
            if text and query.lower() in text.lower():
                snippet = extract_text_snippet(text, query)
                snippets.append({
                    'type': 'audio_transcript',
                    'text': snippet,
                    'confidence': result['transcript'].get('confidence', 0)
                })
        
        elif file_type == 'video' and 'audio_analysis' in result:
            if 'transcript' in result['audio_analysis']:
                text = result['audio_analysis']['transcript'].get('text', '')
                if text and query.lower() in text.lower():
                    snippet = extract_text_snippet(text, query)
                    snippets.append({
                        'type': 'video_transcript',
                        'text': snippet,
                        'confidence': result['audio_analysis']['transcript'].get('confidence', 0)
                    })
        
        elif file_type == 'image' and 'ocr_results' in result:
            text = result['ocr_results'].get('combined_text', '')
            if text and query.lower() in text.lower():
                snippet = extract_text_snippet(text, query)
                snippets.append({
                    'type': 'image_ocr',
                    'text': snippet,
                    'confidence': result['ocr_results'].get('total_confidence', 0) / 100
                })
        
        elif file_type == 'document' and 'content' in result:
            text = result.get('content', '')
            if text and query.lower() in text.lower():
                snippet = extract_text_snippet(text, query)
                snippets.append({
                    'type': 'document_content',
                    'text': snippet,
                    'confidence': 0.9
                })
    
    except Exception as e:
        logger.warning(f"Error extracting snippets: {e}")
    
    return snippets

def extract_text_snippet(text: str, query: str, context_length: int = 150) -> str:
    """Extract snippet around query match"""
    query_lower = query.lower()
    text_lower = text.lower()
    
    pos = text_lower.find(query_lower)
    if pos == -1:
        return text[:context_length] + "..." if len(text) > context_length else text
    
    start = max(0, pos - context_length // 2)
    end = min(len(text), pos + len(query) + context_length // 2)
    
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    
    return snippet

async def generate_content_summary(result: Dict) -> str:
    """Generate summary of multimedia content"""
    file_type = result.get('file_type', 'document')
    
    try:
        if file_type == 'audio':
            duration = result.get('metadata', {}).get('duration', 0)
            word_count = len(result.get('transcript', {}).get('text', '').split())
            return f"Audio: {duration:.1f}s, {word_count} words transcribed"
        
        elif file_type == 'video':
            metadata = result.get('metadata', {})
            duration = metadata.get('duration', 0)
            resolution = metadata.get('resolution', 'Unknown')
            return f"Video: {duration:.1f}s, {resolution}"
        
        elif file_type == 'image':
            metadata = result.get('metadata', {})
            resolution = metadata.get('resolution', 'Unknown')
            image_type = metadata.get('image_type', 'image')
            return f"Image: {resolution}, {image_type}"
        
        else:
            word_count = len(result.get('content', '').split())
            return f"Document: {word_count} words"
    
    except Exception as e:
        logger.warning(f"Error generating content summary: {e}")
        return f"{file_type.title()} file"

# Enhanced WebSocket endpoint for multimedia chat
@app.websocket("/ws/chat/{case_id}")
async def websocket_chat(websocket: WebSocket, case_id: str):
    await manager.connect(websocket, case_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "")
            chat_type = message_data.get("type", "general")
            
            logger.info(f"Received multimedia chat message for case {case_id}: {user_message}")
            
            try:
                # Use enhanced multimedia RAG system
                response = await rag_system.chat_with_case_multimedia(
                    case_id=case_id,
                    user_message=user_message,
                    chat_type=chat_type
                )
                
                # Send enhanced response
                await manager.send_personal_message({
                    "type": "response",
                    "message": response["answer"],
                    "sources": response.get("sources", []),
                    "file_sources": response.get("file_sources", []),
                    "confidence": response.get("confidence", 0.0),
                    "multimedia_analysis": response.get("multimedia_analysis", {}),
                    "context_used": response.get("context_used", False)
                }, websocket)
                
            except Exception as e:
                logger.error(f"Error processing multimedia chat message: {e}", exc_info=True)
                await manager.send_personal_message({
                    "type": "error",
                    "message": "I apologize, but I encountered an error processing your multimedia content. Please try rephrasing your question."
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, case_id)

@app.get("/api/cases/{case_id}/files/{file_id}/analysis")
async def get_file_analysis(case_id: str, file_id: str):
    """Get detailed multimedia analysis for a specific file"""
    try:
        # Search vector store for the specific document
        results = await vector_store.search_documents(case_id, f"id:{file_id}", limit=1)
        
        if not results:
            # Try alternative search by filename
            filename = file_id.replace('_', ' ')
            results = await vector_store.search_documents(case_id, filename, limit=5)
            
            if not results:
                raise HTTPException(status_code=404, detail="File analysis not found")
        
        # Return the most relevant result with enhanced formatting
        analysis = results[0]
        
        # Add multimedia-specific formatting
        enhanced_analysis = await enhance_analysis_display(analysis)
        
        return enhanced_analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving file analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def enhance_analysis_display(analysis: Dict) -> Dict:
    """Enhance analysis data for better display"""
    file_type = analysis.get('file_type', 'document')
    
    try:
        if file_type == 'audio' and 'transcript' in analysis:
            # Format transcript with paragraphs and timestamps
            if 'segments' in analysis['transcript']:
                formatted_transcript = format_transcript_segments(analysis['transcript']['segments'])
                analysis['transcript']['formatted_text'] = formatted_transcript
        
        elif file_type == 'video' and 'audio_analysis' in analysis:
            if 'transcript' in analysis['audio_analysis'] and 'segments' in analysis['audio_analysis']['transcript']:
                formatted_transcript = format_transcript_segments(analysis['audio_analysis']['transcript']['segments'])
                analysis['audio_analysis']['transcript']['formatted_text'] = formatted_transcript
        
        elif file_type == 'image':
            # Add confidence indicators for OCR results
            if 'ocr_results' in analysis:
                analysis['ocr_results']['quality_indicator'] = get_ocr_quality_indicator(
                    analysis['ocr_results'].get('total_confidence', 0)
                )
    
    except Exception as e:
        logger.warning(f"Error enhancing analysis display: {e}")
    
    return analysis

def format_transcript_segments(segments: List[Dict]) -> str:
    """Format transcript segments with timestamps"""
    formatted_lines = []
    
    for segment in segments:
        start_time = segment.get('start', 0)
        text = segment.get('text', '').strip()
        
        if text:
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            formatted_lines.append(f"{timestamp} {text}")
    
    return '\n\n'.join(formatted_lines)

def get_ocr_quality_indicator(confidence: float) -> str:
    """Get quality indicator for OCR confidence"""
    if confidence > 80:
        return 'High'
    elif confidence > 60:
        return 'Medium'
    elif confidence > 40:
        return 'Low'
    else:
        return 'Very Low'

# Processing status endpoint
@app.get("/api/cases/{case_id}/processing-status")
async def get_processing_status(case_id: str):
    """Get processing status for all files in a case"""
    try:
        case_dir = os.path.join(settings.upload_directory, case_id)
        if not os.path.exists(case_dir):
            raise HTTPException(status_code=404, detail="Case not found")
        
        status_info = {
            'case_id': case_id,
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'pending_files': 0,
            'by_type': {
                'audio': {'total': 0, 'processed': 0},
                'video': {'total': 0, 'processed': 0},
                'image': {'total': 0, 'processed': 0},
                'document': {'total': 0, 'processed': 0}
            }
        }
        
        for root, dirs, filenames in os.walk(case_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                file_type = get_file_type(file_path)
                processing_status = await check_file_processing_status(file_path, case_id)
                
                status_info['total_files'] += 1
                status_info['by_type'][file_type]['total'] += 1
                
                if processing_status == 'completed':
                    status_info['processed_files'] += 1
                    status_info['by_type'][file_type]['processed'] += 1
                elif processing_status == 'failed':
                    status_info['failed_files'] += 1
                else:
                    status_info['pending_files'] += 1
        
        return status_info
        
    except Exception as e:
        logger.error(f"Error getting processing status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check with multimedia capabilities
@app.get("/api/health")
async def health_check():
    """Enhanced health check with multimedia capabilities"""
    try:
        # Check core components
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "3.0.0",
            "capabilities": {
                "document_processing": True,
                "audio_processing": document_processor.audio_processor.whisper_model is not None,
                "video_processing": len(document_processor.video_processor.backends) > 0,
                "image_processing": len(document_processor.image_processor.ocr_engines) > 0,
                "vector_search": True,
                "multimedia_chat": True
            },
            "backend_info": {
                "audio_backends": ["whisper"],
                "video_backends": list(document_processor.video_processor.backends.keys()),
                "ocr_backends": list(document_processor.image_processor.ocr_engines.keys())
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)