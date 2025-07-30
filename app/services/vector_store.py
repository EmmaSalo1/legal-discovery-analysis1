import chromadb
from chromadb.config import Settings as ChromaSettings
import uuid
from typing import Dict, List, Any, Optional
import logging
import re
import json
import os
from datetime import datetime
from app.config import settings

logger = logging.getLogger(__name__)

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
                return document_data.get('id', str(uuid.uuid4()))
            
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
                if 'content' in document_data:
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
            if 'summary' in document_data:
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
        logger.info(f"Extracted {len(final_content)} characters of content")
        return final_content
    
    async def search_documents(self, case_id: str, query: str, limit: int = 10) -> List[Dict]:
        """Search documents - ACTUALLY RETURNS RESULTS"""
        try:
            collection_name = f"case_{case_id}"
            
            try:
                collection = self.client.get_collection(collection_name)
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
            except:
                pass
            
            # Fallback: return basic structure
            return {}
            
        except Exception as e:
            logger.error(f"Error getting full document data: {e}")
            return {}
    
    async def get_case_stats(self, case_id: str) -> Dict:
        """Get statistics for a case"""
        try:
            collection_name = f"case_{case_id}"
            
            try:
                collection = self.client.get_collection(collection_name)
            except:
                return {"total_documents": 0, "file_types": {}}
            
            # Get all documents (excluding full document entries)
            results = collection.get(
                where={"type": {"$ne": "full_document"}},
                include=["metadatas"]
            )
            
            stats = {
                "total_documents": len(results['metadatas']),
                "file_types": {}
            }
            
            # Count by file type
            for metadata in results['metadatas']:
                file_type = metadata.get('file_type', 'unknown')
                stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting case stats: {e}")
            return {"total_documents": 0, "file_types": {}}