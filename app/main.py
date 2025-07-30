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
from app.utils.database import get_db, engine, Base

# Import our working vector store
import chromadb
from chromadb.config import Settings as ChromaSettings
import re

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
os.makedirs(settings.chroma_persist_directory, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# WORKING VECTOR STORE CLASS
class WorkingVectorStore:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            logger.info(f"ChromaDB initialized at {settings.chroma_persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
        
    async def add_document(self, document_data: Dict, case_id: str) -> str:
        """Add document to vector store - ACTUALLY WORKS"""
        try:
            logger.info(f"Adding document to vector store for case {case_id}")
            
            # Get or create collection for the case
            collection_name = f"case_{case_id}"
            try:
                collection = self.client.get_collection(collection_name)
                logger.info(f"Using existing collection: {collection_name}")
            except:
                collection = self.client.create_collection(collection_name)
                logger.info(f"Created new collection: {collection_name}")
            
            # Extract searchable content from the document
            content = self._extract_content_from_document(document_data)
            
            if not content.strip():
                logger.warning(f"No content extracted from document {document_data.get('id', 'unknown')}")
                # Still store it with minimal content
                content = f"File: {document_data.get('metadata', {}).get('filename', 'unknown')}"
            
            # Create document ID
            doc_id = document_data.get('id', str(uuid.uuid4()))
            
            # Prepare metadata - convert everything to strings
            metadata = {
                "case_id": str(case_id),
                "file_type": str(document_data.get('file_type', 'document')),
                "filename": str(document_data.get('metadata', {}).get('filename', 'Unknown')),
                "file_path": str(document_data.get('file_path', '')),
                "processing_status": str(document_data.get('processing_status', 'completed')),
                "processed_at": str(document_data.get('processed_at', datetime.now().isoformat())),
                "summary": str(document_data.get('summary', ''))[:500],  # Limit summary length
            }
            
            # Add content length info
            metadata["content_length"] = str(len(content))
            metadata["word_count"] = str(len(content.split()))
            
            # Store the document
            collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Successfully added document {doc_id} to collection {collection_name}")
            
            # Also store the full document data separately for retrieval
            try:
                full_doc_id = f"{doc_id}_full"
                full_metadata = dict(metadata)
                full_metadata["type"] = "full_document"
                full_metadata["original_id"] = doc_id
                
                # Store document data as JSON string
                full_content = json.dumps(document_data, default=str)
                
                collection.add(
                    documents=[full_content],
                    metadatas=[full_metadata],
                    ids=[full_doc_id]
                )
                logger.info(f"Stored full document data for {doc_id}")
            except Exception as e:
                logger.warning(f"Could not store full document data: {e}")
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            raise
    
    def _extract_content_from_document(self, document_data: Dict) -> str:
        """Extract all searchable content from document"""
        content_parts = []
        
        try:
            # Add filename for search
            if 'metadata' in document_data and 'filename' in document_data['metadata']:
                filename = document_data['metadata']['filename']
                content_parts.append(f"FILENAME: {filename}")
            
            # Extract based on file type
            file_type = document_data.get('file_type', 'document')
            
            if file_type == 'document':
                # Regular document content
                if 'content' in document_data and document_data['content']:
                    content_parts.append(f"CONTENT: {document_data['content']}")
            
            elif file_type == 'audio':
                # Audio transcript
                if 'transcript' in document_data and 'text' in document_data['transcript']:
                    content_parts.append(f"TRANSCRIPT: {document_data['transcript']['text']}")
            
            elif file_type == 'video':
                # Video audio transcript
                if 'audio_analysis' in document_data:
                    if 'transcript' in document_data['audio_analysis']:
                        transcript_text = document_data['audio_analysis']['transcript'].get('text', '')
                        if transcript_text:
                            content_parts.append(f"VIDEO_TRANSCRIPT: {transcript_text}")
            
            elif file_type == 'image':
                # Image OCR text
                if 'ocr_results' in document_data:
                    ocr_text = document_data['ocr_results'].get('combined_text', '')
                    if ocr_text:
                        content_parts.append(f"OCR_TEXT: {ocr_text}")
            
            # Add summary
            if 'summary' in document_data and document_data['summary']:
                content_parts.append(f"SUMMARY: {document_data['summary']}")
            
            # Add entities as searchable text
            if 'entities' in document_data:
                entities = document_data['entities']
                for entity_type, items in entities.items():
                    if items and len(items) > 0:
                        items_str = ', '.join(str(item) for item in items[:10])  # Limit to 10 items
                        content_parts.append(f"{entity_type.upper()}: {items_str}")
            
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
        
        final_content = "\n\n".join(content_parts)
        logger.info(f"Extracted {len(final_content)} characters of content for {document_data.get('file_type', 'unknown')} file")
        return final_content
    
    async def search_documents(self, case_id: str, query: str, limit: int = 10) -> List[Dict]:
        """Search documents - ACTUALLY RETURNS RESULTS"""
        try:
            collection_name = f"case_{case_id}"
            
            try:
                collection = self.client.get_collection(collection_name)
                logger.info(f"Searching in collection: {collection_name}")
            except:
                logger.warning(f"Collection {collection_name} not found")
                return []
            
            # Perform the search
            results = collection.query(
                query_texts=[query],
                n_results=min(limit, 50),
                include=["documents", "metadatas", "distances"],
                where={"type": {"$ne": "full_document"}}  # Exclude full document entries
            )
            
            if not results['documents'] or not results['documents'][0]:
                logger.info(f"No search results found for query: {query}")
                return []
            
            search_results = []
            
            for i in range(len(results['documents'][0])):
                try:
                    doc_content = results['documents'][0][i]
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i] if 'distances' in results else 0
                    
                    # Calculate similarity score (1 - distance)
                    similarity_score = max(0, 1 - distance)
                    
                    # Get the full document data
                    full_doc_data = await self._get_full_document_data(collection, results['ids'][0][i])
                    
                    # Create result object
                    result = {
                        'id': results['ids'][0][i],
                        'content': doc_content,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'file_type': metadata.get('file_type', 'document'),
                        'file_path': metadata.get('file_path', ''),
                        'filename': metadata.get('filename', 'Unknown'),
                        'summary': metadata.get('summary', ''),
                        'case_id': case_id
                    }
                    
                    # Add full document data if available
                    if full_doc_data:
                        result.update(full_doc_data)
                    
                    search_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing search result {i}: {e}")
                    continue
            
            logger.info(f"Returning {len(search_results)} search results for case {case_id}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    async def _get_full_document_data(self, collection, doc_id: str) -> Dict:
        """Get full document data by ID"""
        try:
            # Try to get full document data
            full_doc_id = f"{doc_id}_full"
            
            try:
                full_results = collection.get(
                    ids=[full_doc_id],
                    include=["documents", "metadatas"]
                )
                
                if full_results['documents'] and len(full_results['documents']) > 0:
                    doc_json = full_results['documents'][0]
                    return json.loads(doc_json)
            except Exception as e:
                logger.debug(f"Could not get full document data: {e}")
                pass
            
            # Fallback: return basic structure
            return {}
            
        except Exception as e:
            logger.error(f"Error getting full document data: {e}")
            return {}

# WORKING RAG SYSTEM
# REPLACE the WorkingRAGSystem class in your main.py with this:

class WorkingRAGSystem:
    # Add this to the top of your WorkingRAGSystem.__init__ method to debug OpenAI:

    def __init__(self, vector_store):
        self.vector_store = vector_store
        try:
            from openai import AsyncOpenAI
        
        # Debug OpenAI configuration
            api_key = settings.openai_api_key
            logger.info(f"OpenAI API key configured: {api_key[:10] if api_key and len(api_key) > 10 else 'NO KEY'}...")
            logger.info(f"OpenAI model: {settings.openai_model}")
        
            if not api_key or api_key == "your_openai_api_key_here":
                logger.error("OpenAI API key not properly configured!")
                self.client = None
            else:
                self.client = AsyncOpenAI(api_key=api_key)
                logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    async def chat_with_case(self, case_id: str, user_message: str, chat_type: str = "general") -> Dict:
        """Chat with case - ACTUALLY WORKS WITH UPLOADED FILES"""
        try:
            logger.info(f"Processing chat for case {case_id}: {user_message}")
            
            # Search for documents
            search_results = await self.vector_store.search_documents(case_id, user_message, limit=10)
            logger.info(f"Found {len(search_results)} search results")
            
            # Debug: Log what we found
            for i, result in enumerate(search_results):
                logger.info(f"Result {i}: filename={result.get('filename', 'unknown')}, similarity={result.get('similarity_score', 0)}, content_length={len(result.get('content', ''))}")
            
            # Build context from search results - LOWERED THRESHOLD
            context_parts = []
            file_sources = []
            
            for result in search_results[:5]:  # Use top 5 results
                try:
                    content = result.get('content', '')
                    filename = result.get('filename', 'Unknown')
                    file_type = result.get('file_type', 'document')
                    similarity = result.get('similarity_score', 0)
                    
                    # MUCH LOWER threshold - accept almost anything with content
                    if content and len(content.strip()) > 10:  # Just need some content
                        # Extract relevant excerpt
                        excerpt = self._extract_relevant_excerpt(content, user_message)
                        
                        if excerpt and len(excerpt.strip()) > 10:
                            context_parts.append(f"From {filename}:\n{excerpt}")
                            file_sources.append({
                                'name': filename,
                                'type': file_type,
                                'id': result.get('id', 'unknown'),
                                'confidence': max(0.5, float(similarity)),  # Boost confidence
                                'relevance_score': max(0.5, float(similarity)),  # Boost relevance
                                'path': result.get('file_path', ''),
                                'content_type': 'extracted_content'
                            })
                            logger.info(f"Added {filename} to context (similarity: {similarity})")
                        
                except Exception as e:
                    logger.error(f"Error processing search result: {e}")
                    continue
            
            logger.info(f"Built context from {len(context_parts)} parts, {len(file_sources)} file sources")
            
            # Generate response
            if self.client and context_parts:
                try:
                    context_text = "\n\n".join(context_parts)
                    
                    system_prompt = """You are a helpful legal assistant analyzing case documents. 
                    Provide clear, accurate answers based on the provided document excerpts.
                    Always reference which documents your answer comes from.
                    Be specific about what information you found in each document."""
                    
                    user_prompt = f"Question: {user_message}\n\nRelevant document excerpts:\n\n{context_text}"
                    
                    logger.info(f"Sending to OpenAI: {len(user_prompt)} characters")
                    
                    response = await self.client.chat.completions.create(
                        model=settings.openai_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=500
                    )
                    
                    answer = response.choices[0].message.content
                    logger.info("Got response from OpenAI")
                    
                except Exception as e:
                    logger.error(f"OpenAI API call failed: {e}")
                    answer = f"I found relevant information in {len(file_sources)} files, but couldn't generate a response due to an API error. The files contain information related to your query about: {user_message}"
            
            elif context_parts:
                # Fallback without OpenAI - show what we found
                answer = f"I found information in {len(file_sources)} files:\n\n"
                for source in file_sources:
                    answer += f"• **{source['name']}** ({source['type']})\n"
                
                answer += f"\nThese files contain content related to your query: '{user_message}'. "
                
                # Show a snippet of what we found
                if context_parts:
                    answer += f"\n\nHere's what I found:\n\n{context_parts[0][:500]}..."
            
            elif search_results:
                # We found results but no good content
                answer = f"I found {len(search_results)} files that might be relevant, but I couldn't extract useful content from them. Files found:\n\n"
                for result in search_results[:3]:
                    answer += f"• {result.get('filename', 'unknown')} (similarity: {result.get('similarity_score', 0):.2f})\n"
                
                answer += "\nThe files may need more processing time or may not contain text content."
            
            else:
                # No results at all
                answer = f"I couldn't find any files in case '{case_id}' that match your query: '{user_message}'. Make sure files have been uploaded and processed."
            
            return {
                "answer": answer,
                "sources": [source.get('path', source.get('name', '')) for source in file_sources],
                "file_sources": file_sources,
                "confidence": 0.8 if file_sources else 0.2,
                "context_used": len(context_parts) > 0
            }
            
        except Exception as e:
            logger.error(f"Error in chat_with_case: {e}")
            return {
                "answer": f"I encountered an error processing your request: {str(e)}. Please try again.",
                "sources": [],
                "file_sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _extract_relevant_excerpt(self, content: str, query: str) -> str:
        """Extract relevant excerpt from content"""
        if not content:
            return ""
        
        # If content is short enough, return it all
        if len(content) <= 400:
            return content
        
        if not query:
            return content[:400] + "..."
        
        # Simple excerpt extraction
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        content_lower = content.lower()
        
        # Find the best position in the content
        best_pos = 0
        best_score = 0
        
        # Try to find where query words appear
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1:
                # Check context around this word
                start = max(0, pos - 100)
                end = min(len(content), pos + 300)
                chunk = content_lower[start:end]
                
                score = sum(1 for qword in query_words if qword in chunk)
                if score > best_score:
                    best_score = score
                    best_pos = start
                    break
        
        # Extract excerpt around best position
        start = max(0, best_pos)
        end = min(len(content), best_pos + 400)
        excerpt = content[start:end]
        
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(content):
            excerpt = excerpt + "..."
        
        return excerpt


# Initialize services
document_processor = DocumentProcessor()
vector_store = WorkingVectorStore()
rag_system = WorkingRAGSystem(vector_store)

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
            logger.error(f"Error sending WebSocket message: {e}")

    async def broadcast_to_case(self, message: dict, case_id: str):
        """Broadcast message to all connections for a specific case"""
        if case_id not in self.case_connections:
            return
        
        connections = self.case_connections[case_id].copy()
        
        for connection in connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to case {case_id}: {e}")
                self.disconnect(connection, case_id)

manager = ConnectionManager()

# Utility functions
def get_file_type(file_path: str) -> str:
    """Determine file type"""
    mime_type, _ = mimetypes.guess_type(file_path)
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension in settings.supported_audio_formats:
        return 'audio'
    elif extension in settings.supported_video_formats:
        return 'video'
    elif extension in settings.supported_image_formats:
        return 'image'
    else:
        return 'document'

async def process_file_background(file_path: str, case_id: str, file_type: str, processing_id: str):
    """Background processing for files"""
    try:
        logger.info(f"Starting background processing for {file_path}")
        
        # Process the file
        result = await document_processor.process_document(file_path, case_id)
        
        # Add to vector store if successful
        if result.get('processing_status') == 'completed':
            await vector_store.add_document(result, case_id)
            logger.info(f"Successfully processed and added {file_path} to vector store")
        else:
            logger.warning(f"Processing failed for {file_path}: {result.get('error', 'Unknown error')}")
        
        # Notify completion via WebSocket
        await manager.broadcast_to_case({
            'type': 'processing_complete',
            'processing_id': processing_id,
            'status': 'completed' if result.get('processing_status') == 'completed' else 'failed',
            'result': result
        }, case_id)
        
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
            
            # Determine subdirectory
            if file_type in ['audio', 'video', 'image']:
                subdir = file_type
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
            
            # Queue background processing
            background_tasks.add_task(
                process_file_background,
                file_path, case_id, file_type, processing_id
            )
            
            logger.info(f"Uploaded and queued for processing: {file.filename}")
        
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
            
            # Extract relevant snippets
            content = result.get('content', '')
            if query.lower() in content.lower():
                # Simple snippet extraction
                start = max(0, content.lower().find(query.lower()) - 50)
                end = min(len(content), start + 200)
                snippet = content[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(content):
                    snippet = snippet + "..."
                snippets.append({'type': 'content', 'text': snippet})
            
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
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving file analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for chat
@app.websocket("/ws/chat/{case_id}")
async def websocket_chat(websocket: WebSocket, case_id: str):
    await manager.connect(websocket, case_id)
    logger.info(f"WebSocket connected for case {case_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "")
            chat_type = message_data.get("type", "general")
            
            logger.info(f"Received chat message for case {case_id}: {user_message}")
            
            try:
                # Get AI response using our working RAG system
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
                
                logger.info(f"Sent response for case {case_id} with {len(response.get('file_sources', []))} sources")
                
            except Exception as e:
                logger.error(f"Error processing chat message: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "I apologize, but I encountered an error processing your request. Please try again."
                }, websocket)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for case {case_id}")
        manager.disconnect(websocket, case_id)
    except Exception as e:
        logger.error(f"WebSocket error for case {case_id}: {e}")
        manager.disconnect(websocket, case_id)

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