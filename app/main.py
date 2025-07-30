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
from app.services.document_processor import DocumentProcessor
from app.services.rag_system import RAGSystem
from app.services.vector_store import VectorStore
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
    version="2.0.0",
    description="Advanced legal discovery system with multimedia support"
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

# Initialize services
document_processor = DocumentProcessor()
vector_store = VectorStore()
rag_system = RAGSystem(vector_store)

# WebSocket connection manager
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

    def disconnect(self, websocket: WebSocket, case_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if case_id in self.case_connections and websocket in self.case_connections[case_id]:
            self.case_connections[case_id].remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def broadcast_to_case(self, message: dict, case_id: str):
        """Fixed broadcast method"""
        if case_id in self.case_connections:
            disconnected = []
            for connection in self.case_connections[case_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to case {case_id}: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                if conn in self.case_connections[case_id]:
                    self.case_connections[case_id].remove(conn)
                if conn in self.active_connections:
                    self.active_connections.remove(conn)

manager = ConnectionManager()

# Utility functions
def get_file_type(file_path: str) -> str:
    """Determine file type with better logic"""
    mime_type, _ = mimetypes.guess_type(file_path)
    extension = os.path.splitext(file_path)[1].lower()
    
    # Audio files
    if extension in settings.supported_audio_formats or (mime_type and mime_type.startswith('audio/')):
        return 'audio'
    # Video files
    elif extension in settings.supported_video_formats or (mime_type and mime_type.startswith('video/')):
        return 'video'
    # Image files (but NOT PDFs)
    elif extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'] or (mime_type and mime_type.startswith('image/')):
        return 'image'
    # Everything else is a document (including PDFs)
    else:
        return 'document'

async def process_file_background(file_path: str, case_id: str, file_type: str, processing_id: str):
    """Background processing for files with better error handling"""
    try:
        logger.info(f"Starting background processing for {file_path}")
        
        # Process the file
        result = await document_processor.process_document(file_path, case_id)
        
        # Add to vector store if successful
        if result.get('processing_status') == 'completed':
            doc_id = await vector_store.add_document(result, case_id)
            logger.info(f"Added document to vector store with ID: {doc_id}")
        
        # Notify completion via WebSocket
        await manager.broadcast_to_case({
            'type': 'processing_complete',
            'processing_id': processing_id,
            'status': 'completed' if result.get('processing_status') == 'completed' else 'failed',
            'result': result
        }, case_id)
        
        logger.info(f"Completed processing for {file_path}")
        
    except Exception as e:
        logger.error(f"Background processing failed for {file_path}: {str(e)}")
        await manager.broadcast_to_case({
            'type': 'processing_complete',
            'processing_id': processing_id,
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
    """Create a new case"""
    try:
        # Create case directory
        case_dir = os.path.join(settings.upload_directory, case.case_number)
        os.makedirs(case_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "plaintiff_production/documents",
            "plaintiff_production/audio", 
            "plaintiff_production/video",
            "plaintiff_production/images",
            "defendant_production/documents",
            "defendant_production/audio",
            "defendant_production/video", 
            "defendant_production/images",
            "third_party_production",
            "court_filings"
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(case_dir, subdir), exist_ok=True)
        
        logger.info(f"Created case: {case.case_number}")
        return {"id": case.case_number, **case.dict()}
        
    except Exception as e:
        logger.error(f"Error creating case: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cases/{case_id}/files")
async def list_case_files(case_id: str, file_type: Optional[str] = None):
    """List files in a case"""
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
                
                file_info = {
                    'filename': filename,
                    'path': relative_path,
                    'type': detected_type,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                }
                
                files.append(file_info)
        
        return {'case_id': case_id, 'files': files, 'total_files': len(files)}
        
    except Exception as e:
        logger.error(f"Error listing case files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cases/{case_id}/documents/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    case_id: str,
    files: List[UploadFile] = File(...),
    document_type: str = Form("general"),
    party: str = Form("plaintiff_production")
):
    """Upload documents to a case"""
    try:
        uploaded_files = []
        
        for file in files:
            processing_id = str(uuid.uuid4())
            file_type = get_file_type(file.filename)
            
            logger.info(f"Processing upload: {file.filename}, type: {file_type}")

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
            
            # Save file
            file_path = os.path.join(file_dir, file.filename)
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            uploaded_files.append({
                'filename': file.filename,
                'file_path': file_path,
                'file_type': file_type,
                'processing_id': processing_id,
                'size': len(content)
            })
            logger.info(f"Queuing background task for {file.filename}")

            # Queue background processing
            await process_file_background(file_path, case_id, file_type, processing_id)

            
            logger.info(f"Background task queued for {file.filename}")

            logger.info(f"Queued processing for {file.filename} (type: {file_type})")
        
        return {
            "message": f"Uploaded {len(uploaded_files)} files",
            "files": uploaded_files
        }
        
    except Exception as e:
        logger.error(f"Error uploading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cases/{case_id}/search")
async def search_case(case_id: str, query: str, file_types: Optional[str] = None, limit: int = 50):
    """Search case documents"""
    try:
        # Search vector store
        results = await vector_store.search_documents(case_id, query, limit=limit)
        
        # Enhance results with snippets
        enhanced_results = []
        for result in results:
            snippets = []
            
            # Extract relevant snippets from different content types
            if result.get('file_type') == 'audio' and result.get('transcript', {}).get('text'):
                content = result['transcript']['text']
                snippet = extract_snippet(content, query)
                if snippet:
                    snippets.append({'type': 'transcript', 'text': snippet})
                    
            elif result.get('file_type') == 'video' and result.get('audio_analysis', {}).get('transcript', {}).get('text'):
                content = result['audio_analysis']['transcript']['text']
                snippet = extract_snippet(content, query)
                if snippet:
                    snippets.append({'type': 'video_transcript', 'text': snippet})
                    
            elif result.get('file_type') == 'image' and result.get('ocr_results', {}).get('combined_text'):
                content = result['ocr_results']['combined_text']
                snippet = extract_snippet(content, query)
                if snippet:
                    snippets.append({'type': 'ocr_text', 'text': snippet})
                    
            elif result.get('file_type') == 'document' and result.get('content'):
                content = result['content']
                snippet = extract_snippet(content, query)
                if snippet:
                    snippets.append({'type': 'document_content', 'text': snippet})
            
            enhanced_result = {
                **result,
                'snippets': snippets,
                'relevance_score': result.get('similarity_score', 0)
            }
            enhanced_results.append(enhanced_result)
        
        return {
            'case_id': case_id,
            'query': query,
            'results': enhanced_results,
            'total_results': len(enhanced_results)
        }
        
    except Exception as e:
        logger.error(f"Error searching case: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def extract_snippet(content: str, query: str, context_length: int = 150) -> str:
    """Extract relevant snippet from content"""
    if not content or not query:
        return ""
    
    content_lower = content.lower()
    query_lower = query.lower()
    
    # Find query in content
    pos = content_lower.find(query_lower)
    if pos == -1:
        # If exact match not found, return first part of content
        return content[:context_length] + "..." if len(content) > context_length else content
    
    # Extract context around the query
    start = max(0, pos - context_length // 2)
    end = min(len(content), pos + len(query) + context_length // 2)
    
    snippet = content[start:end]
    
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    
    return snippet

# WebSocket endpoint for chat
@app.websocket("/ws/chat/{case_id}")
async def websocket_chat(websocket: WebSocket, case_id: str):
    await manager.connect(websocket, case_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "")
            chat_type = message_data.get("type", "general")
            
            logger.info(f"Received chat message for case {case_id}: {user_message}")
            
            try:
                # Get AI response
                response = await rag_system.chat_with_case(
                    case_id=case_id,
                    user_message=user_message,
                    chat_type=chat_type
                )
                
                # Send response
                await manager.send_personal_message({
                    "type": "response",
                    "message": response["answer"],
                    "sources": response.get("sources", []),
                    "file_sources": response.get("file_sources", []),
                    "confidence": response.get("confidence", 0.0)
                }, websocket)
                
            except Exception as e:
                logger.error(f"Error processing chat message: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "I apologize, but I encountered an error processing your request. Please try rephrasing your question."
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, case_id)

@app.get("/api/cases/{case_id}/files/{file_id}/analysis")
async def get_file_analysis(case_id: str, file_id: str):
    """Get detailed analysis for a specific file"""
    try:
        # Search vector store for the specific document
        results = await vector_store.search_documents(case_id, f"id:{file_id}", limit=1)
        
        if not results:
            # Try alternative search by filename
            filename = file_id.replace('_', ' ')
            results = await vector_store.search_documents(case_id, filename, limit=5)
            
            if not results:
                raise HTTPException(status_code=404, detail="File analysis not found")
        
        # Return the most relevant result
        analysis = results[0]
        
        # Add some additional formatting for better display
        if 'transcript' in analysis and 'text' in analysis['transcript']:
            # Format transcript with paragraphs
            text = analysis['transcript']['text']
            # Split into sentences and group them
            sentences = text.split('. ')
            paragraphs = []
            current_paragraph = []
            
            for sentence in sentences:
                current_paragraph.append(sentence)
                if len(current_paragraph) >= 3:  # Group every 3 sentences
                    paragraphs.append('. '.join(current_paragraph) + '.')
                    current_paragraph = []
            
            if current_paragraph:
                paragraphs.append('. '.join(current_paragraph) + '.')
            
            analysis['transcript']['formatted_text'] = '\n\n'.join(paragraphs)
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving file analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)