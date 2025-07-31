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

class EnhancedVectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        logger.info("Enhanced Vector Store initialized with multimedia support")
        
    async def add_document(self, document_data: Dict, case_id: str) -> str:
        """Add document to vector store with comprehensive multimedia support"""
        try:
            # Get or create collection for the case
            collection_name = f"case_{case_id}"
            collection = self._get_or_create_collection(collection_name)
            
            # Extract text content based on document type
            text_content = self._extract_comprehensive_text_content(document_data)
            
            if not text_content:
                logger.warning(f"No text content found for document {document_data.get('id', 'unknown')}")
                return document_data.get('id', str(uuid.uuid4()))
            
            # Create enhanced metadata with multimedia support
            metadata = self._create_multimedia_metadata(document_data, case_id)
            
            doc_id = document_data.get('id', str(uuid.uuid4()))
            
            # Add to collection with chunking for large documents
            chunks = self._intelligent_chunk_document(text_content, document_data.get('file_type', 'document'))
            
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_metadata = {**metadata, 'chunk_index': i, 'total_chunks': len(chunks)}
                
                collection.add(
                    documents=[chunk],
                    metadatas=[chunk_metadata],
                    ids=[chunk_id]
                )
                chunk_ids.append(chunk_id)
            
            # Store the full document data separately for retrieval
            await self._store_full_document_data(collection, doc_id, document_data)
            
            logger.info(f"Added document {doc_id} to vector store with {len(chunks)} chunks")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            raise
    
    def _extract_comprehensive_text_content(self, document_data: Dict) -> str:
        """Extract searchable text content from various multimedia document types"""
        text_parts = []
        file_type = document_data.get('file_type', 'document')
        
        # Extract filename for search
        filename = document_data.get('metadata', {}).get('filename', '')
        if filename:
            text_parts.append(f"FILENAME: {filename}")
        
        # File type specific extraction
        if file_type == 'audio':
            # Extract audio transcript
            if 'transcript' in document_data:
                transcript_text = document_data['transcript'].get('text', '')
                if transcript_text:
                    text_parts.append(f"TRANSCRIPT: {transcript_text}")
                
                # Add formatted transcript with timestamps
                formatted_transcript = document_data['transcript'].get('formatted_transcript', '')
                if formatted_transcript:
                    text_parts.append(f"TIMED_TRANSCRIPT: {formatted_transcript}")
            
            # Add speaker information
            if 'speaker_analysis' in document_data:
                speaker_segments = document_data['speaker_analysis'].get('speaker_segments', [])
                speaker_texts = []
                for segment in speaker_segments:
                    speaker_id = segment.get('speaker_id', 'Unknown')
                    text = segment.get('text', '')
                    if text:
                        speaker_texts.append(f"{speaker_id}: {text}")
                
                if speaker_texts:
                    text_parts.append(f"SPEAKER_SEGMENTS: {' '.join(speaker_texts)}")
            
            # Add content analysis
            if 'content_analysis' in document_data:
                content_analysis = document_data['content_analysis']
                legal_terms = content_analysis.get('legal_terms', [])
                if legal_terms:
                    text_parts.append(f"LEGAL_TERMS: {', '.join(legal_terms)}")
        
        elif file_type == 'video':
            # Extract video audio transcript
            if 'audio_analysis' in document_data and 'transcript' in document_data['audio_analysis']:
                transcript_text = document_data['audio_analysis']['transcript'].get('text', '')
                if transcript_text:
                    text_parts.append(f"VIDEO_TRANSCRIPT: {transcript_text}")
            
            # Add visual analysis information
            if 'visual_analysis' in document_data:
                visual = document_data['visual_analysis']
                visual_info = []
                
                if visual.get('has_faces'):
                    visual_info.append("contains people")
                if visual.get('has_text'):
                    visual_info.append("contains visible text")
                
                layout_type = visual.get('layout_type', '')
                if layout_type:
                    visual_info.append(f"layout: {layout_type}")
                
                if visual_info:
                    text_parts.append(f"VISUAL_CONTENT: {', '.join(visual_info)}")
        
        elif file_type == 'image':
            # Extract OCR text
            if 'ocr_results' in document_data:
                ocr_text = document_data['ocr_results'].get('combined_text', '')
                if ocr_text:
                    text_parts.append(f"OCR_TEXT: {ocr_text}")
                
                # Add best OCR result
                best_result = document_data['ocr_results'].get('best_result', '')
                if best_result and best_result != ocr_text:
                    text_parts.append(f"OCR_BEST: {best_result}")
            
            # Add document structure analysis
            if 'document_analysis' in document_data:
                doc_analysis = document_data['document_analysis']
                doc_type = doc_analysis.get('document_type', '')
                if doc_type and doc_type != 'unknown':
                    text_parts.append(f"DOCUMENT_TYPE: {doc_type}")
            
            # Add structured data
            if 'structured_data' in document_data:
                structured = document_data['structured_data']
                for data_type, items in structured.items():
                    if items:
                        text_parts.append(f"{data_type.upper()}: {', '.join(map(str, items))}")
        
        elif file_type == 'document':
            # Extract traditional document content
            content = document_data.get('content', '')
            if content and content.strip():
                text_parts.append(f"CONTENT: {content}")
            
            # Add document structure
            if 'document_structure' in document_data:
                structure = document_data['document_structure']
                doc_type = structure.get('document_type', '')
                if doc_type and doc_type != 'unknown':
                    text_parts.append(f"DOCUMENT_TYPE: {doc_type}")
        
        # Universal content additions
        
        # Extract summary
        if 'summary' in document_data:
            text_parts.append(f"SUMMARY: {document_data['summary']}")
        
        # Extract entities (common across all types)
        if 'entities' in document_data:
            entities = document_data['entities']
            entity_text = []
            
            for entity_type, items in entities.items():
                if items:
                    entity_text.append(f"{entity_type.upper()}: {', '.join(map(str, items))}")
            
            if entity_text:
                text_parts.append(f"ENTITIES: {' | '.join(entity_text)}")
        
        # Extract privilege information
        if 'privilege_flags' in document_data and document_data['privilege_flags']:
            privilege_contexts = []
            for flag in document_data['privilege_flags']:
                context = flag.get('context', '')
                if context:
                    privilege_contexts.append(context)
            
            if privilege_contexts:
                text_parts.append(f"PRIVILEGE_CONTEXT: {' '.join(privilege_contexts)}")
        
        combined_text = "\n\n".join