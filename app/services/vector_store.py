import chromadb
import uuid
from typing import Dict, List, Any, Optional
import logging
import re
import json
import os
from datetime import datetime
from app.config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        try:
            # ChromaDB 1.x API
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory
            )
            logger.info(f"Initialized ChromaDB at {settings.chroma_persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def add_document(self, document_data: Dict, case_id: str) -> str:
        """Add document to vector store with multimedia support"""
        try:
            # Get or create collection for the case
            collection_name = f"case_{case_id}"
            collection = self._get_or_create_collection(collection_name)
            
            # Extract text content based on document type
            text_content = self._extract_text_content(document_data)
            
            if not text_content:
                logger.warning(f"No text content found for document {document_data.get('id', 'unknown')}")
                return document_data.get('id', str(uuid.uuid4()))
            
            # Create enhanced metadata with search optimization
            metadata = self._create_enhanced_metadata(document_data, case_id)
            
            doc_id = document_data.get('id', str(uuid.uuid4()))
            
            # Add to collection (ChromaDB 1.x API)
            collection.add(
                documents=[text_content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            # Also store the full document data for retrieval
            self._store_full_document(collection, doc_id, document_data)
            
            logger.info(f"Added document {doc_id} to vector store")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            raise
    
    def _create_enhanced_metadata(self, document_data: Dict, case_id: str) -> Dict:
        """Create enhanced metadata for better search filtering"""
        metadata = {
            "file_path": str(document_data.get('file_path', '')),
            "file_type": str(document_data.get('file_type', 'document')),
            "case_id": str(case_id),
            "processing_status": str(document_data.get('processing_status', 'completed')),
            "filename": str(document_data.get('metadata', {}).get('filename', '')),
        }
        
        # Add content indicators for better filtering
        if document_data.get('file_type') == 'audio':
            metadata.update({
                "duration": str(document_data.get('metadata', {}).get('duration', 0)),
                "has_transcript": str('transcript' in document_data),
                "transcript_confidence": str(document_data.get('transcript', {}).get('confidence', 0)),
                "word_count": str(len(document_data.get('transcript', {}).get('text', '').split()) if document_data.get('transcript', {}).get('text') else 0)
            })
        elif document_data.get('file_type') == 'video':
            metadata.update({
                "duration": str(document_data.get('metadata', {}).get('duration', 0)),
                "resolution": str(document_data.get('metadata', {}).get('resolution', '')),
                "has_audio": str('audio_analysis' in document_data),
                "word_count": str(len(document_data.get('audio_analysis', {}).get('transcript', {}).get('text', '').split()) if document_data.get('audio_analysis', {}).get('transcript', {}).get('text') else 0)
            })
        elif document_data.get('file_type') == 'image':
            metadata.update({
                "resolution": str(document_data.get('metadata', {}).get('resolution', '')),
                "has_ocr": str('ocr_results' in document_data),
                "ocr_confidence": str(document_data.get('ocr_results', {}).get('total_confidence', 0)),
                "word_count": str(len(document_data.get('ocr_results', {}).get('combined_text', '').split()) if document_data.get('ocr_results', {}).get('combined_text') else 0)
            })
        elif document_data.get('file_type') == 'document':
            metadata.update({
                "word_count": str(len(document_data.get('content', '').split()) if document_data.get('content') else 0),
                "has_content": str(bool(document_data.get('content', '').strip()))
            })
        
        return metadata
    
    def _store_full_document(self, collection, doc_id: str, document_data: Dict):
        """Store full document data for retrieval"""
        try:
            # Store full document as separate entry for retrieval
            full_doc_id = f"{doc_id}_full"
            full_doc_metadata = {
                "type": "full_document", 
                "original_id": str(doc_id),
                "case_id": str(document_data.get('case_id', '')),
                "file_type": str(document_data.get('file_type', 'document'))
            }
            
            collection.add(
                documents=[json.dumps(document_data)],
                metadatas=[full_doc_metadata],
                ids=[full_doc_id]
            )
        except Exception as e:
            logger.warning(f"Could not store full document data: {e}")
    
    async def search_documents(self, case_id: str, query: str, limit: int = 10) -> List[Dict]:
        """Enhanced search with better relevance filtering"""
        try:
            collection_name = f"case_{case_id}"
            
            try:
                collection = self.client.get_collection(collection_name)
            except:
                logger.warning(f"Collection {collection_name} not found")
                return []
            
            # Preprocess query for better search
            processed_query = self._preprocess_query(query)
            
            # Perform similarity search (ChromaDB 1.x API)
            results = collection.query(
                query_texts=[processed_query],
                n_results=min(limit * 3, 50),
                include=["documents", "metadatas", "distances"],
                where={"type": {"$ne": "full_document"}}
            )
            
            # Format and filter results
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Skip results that are too distant (not relevant)
                    similarity_score = 1 - distance
                    if similarity_score < 0.2:
                        continue
                    
                    # Get full document data
                    full_doc_data = await self._get_full_document_data(collection, results['ids'][0][i])
                    
                    if full_doc_data:
                        # Add search metadata
                        full_doc_data['similarity_score'] = similarity_score
                        full_doc_data['search_metadata'] = metadata
                        
                        # Filter by content quality
                        if self._has_relevant_content(full_doc_data, query):
                            formatted_results.append(full_doc_data)
            
            # Sort by similarity and content relevance
            formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Return top results
            return formatted_results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better search results"""
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'should', 'would', 'will']
        words = re.findall(r'\w+', query.lower())
        filtered_words = [w for w in words if w not in question_words]
        
        if len(filtered_words) < 2:
            filtered_words = words
        
        return ' '.join(filtered_words)
    
    async def _get_full_document_data(self, collection, doc_id: str) -> Optional[Dict]:
        """Retrieve full document data by ID"""
        try:
            # Try to get from stored full documents first
            full_results = collection.query(
                query_texts=[""],
                n_results=1,
                include=["documents", "metadatas"],
                where={"original_id": doc_id}
            )
            
            if full_results['documents'] and full_results['documents'][0]:
                doc_str = full_results['documents'][0][0]
                try:
                    return json.loads(doc_str)
                except Exception as e:
                    logger.warning(f"Failed to parse stored document data: {e}")
            
            # Fallback: reconstruct from metadata and content
            basic_results = collection.query(
                query_texts=[""],
                n_results=1,
                include=["documents", "metadatas"],
                ids=[doc_id]
            )
            
            if basic_results['documents'] and basic_results['documents'][0]:
                metadata = basic_results['metadatas'][0][0]
                content = basic_results['documents'][0][0]
                
                return {
                    'id': doc_id,
                    'file_path': metadata.get('file_path', ''),
                    'file_type': metadata.get('file_type', 'document'),
                    'content': content,
                    'metadata': metadata
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting full document data: {e}")
            return None
    
    def _has_relevant_content(self, doc_data: Dict, query: str) -> bool:
        """Check if document has relevant content for the query"""
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words = query_words - stop_words
        
        if not query_words:
            return True
        
        # Check content based on file type
        searchable_text = ""
        
        if doc_data.get('file_type') == 'audio' and 'transcript' in doc_data:
            searchable_text = doc_data['transcript'].get('text', '')
        elif doc_data.get('file_type') == 'video' and 'audio_analysis' in doc_data:
            if 'transcript' in doc_data['audio_analysis']:
                searchable_text = doc_data['audio_analysis']['transcript'].get('text', '')
        elif doc_data.get('file_type') == 'image' and 'ocr_results' in doc_data:
            searchable_text = doc_data['ocr_results'].get('combined_text', '')
        elif doc_data.get('file_type') == 'document':
            searchable_text = doc_data.get('content', '')
        
        if not searchable_text:
            return False
        
        searchable_lower = searchable_text.lower()
        content_words = set(re.findall(r'\w+', searchable_lower))
        matches = query_words.intersection(content_words)
        
        return len(matches) > 0
    
    def _extract_text_content(self, document_data: Dict) -> str:
        """Extract searchable text content from various document types"""
        text_parts = []
        
        # Extract filename for search
        filename = document_data.get('metadata', {}).get('filename', '')
        if filename:
            text_parts.append(f"FILENAME: {filename}")
        
        # Extract transcript text (audio/video)
        if 'transcript' in document_data:
            transcript_text = document_data['transcript'].get('text', '')
            if transcript_text:
                text_parts.append(f"TRANSCRIPT: {transcript_text}")
        
        # Extract video audio transcript
        if 'audio_analysis' in document_data and 'transcript' in document_data['audio_analysis']:
            transcript_text = document_data['audio_analysis']['transcript'].get('text', '')
            if transcript_text:
                text_parts.append(f"AUDIO_TRANSCRIPT: {transcript_text}")
        
        # Extract OCR text (images)
        if 'ocr_results' in document_data:
            ocr_text = document_data['ocr_results'].get('combined_text', '')
            if ocr_text:
                text_parts.append(f"OCR_TEXT: {ocr_text}")
        
        # Extract traditional document content
        if 'content' in document_data:
            content = document_data['content']
            if content and content.strip():
                text_parts.append(f"CONTENT: {content}")
        
        # Extract summary
        if 'summary' in document_data:
            text_parts.append(f"SUMMARY: {document_data['summary']}")
        
        # Extract key entities as searchable text
        if 'entities' in document_data:
            entities = document_data['entities']
            entity_text = []
            
            for entity_type, items in entities.items():
                if items:
                    entity_text.append(f"{entity_type.upper()}: {', '.join(map(str, items))}")
            
            if entity_text:
                text_parts.append(f"ENTITIES: {' | '.join(entity_text)}")
        
        return "\n\n".join(text_parts)
    
    def _get_or_create_collection(self, collection_name: str):
        """Get existing collection or create new one"""
        try:
            return self.client.get_collection(collection_name)
        except:
            return self.client.create_collection(
                collection_name,
                metadata={"description": f"Legal discovery collection for {collection_name}"}
            )
    
    async def get_case_documents(self, case_id: str) -> List[Dict]:
        """Get all documents for a case"""
        try:
            collection_name = f"case_{case_id}"
            
            try:
                collection = self.client.get_collection(collection_name)
            except:
                logger.info(f"No collection found for case {case_id}")
                return []
            
            # Get all documents
            results = collection.get(
                limit=1000,
                include=["documents", "metadatas"],
                where={"type": {"$ne": "full_document"}}
            )
            
            documents = {}
            for i, doc_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                document_id = metadata.get("original_id", doc_id)
                
                if document_id not in documents:
                    documents[document_id] = {
                        "id": document_id,
                        "filename": metadata.get("filename", "Unknown"),
                        "case_id": case_id,
                        "file_type": metadata.get("file_type", "document"),
                        "file_path": metadata.get("file_path", ""),
                        "processing_status": metadata.get("processing_status", "unknown"),
                        "metadata": metadata
                    }
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Error getting case documents: {e}")
            return []
    
    def delete_case_collection(self, case_id: str) -> bool:
        """Delete all documents for a case"""
        try:
            collection_name = f"case_{case_id}"
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection for case {case_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting case collection: {e}")
            return False