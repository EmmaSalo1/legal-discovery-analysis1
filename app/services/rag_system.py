from openai import AsyncOpenAI
from typing import Dict, List, Any, Optional
import logging
import os
import re
from app.config import settings
from app.services.vector_store import EnhancedVectorStore

logger = logging.getLogger(__name__)

class EnhancedRAGSystem:
    def __init__(self, vector_store: EnhancedVectorStore):
        self.vector_store = vector_store
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        logger.info("Enhanced RAG System initialized with multimedia support")
        
    async def chat_with_case_multimedia(self, case_id: str, user_message: str, chat_type: str = "general") -> Dict:
        """Enhanced chat system specifically designed for multimedia content"""
        try:
            logger.info(f"Processing multimedia chat for case {case_id}: {user_message}")
            
            # Analyze the question to determine what type of content user is asking about
            question_analysis = self._analyze_question_intent(user_message)
            
            # Search with multimedia-specific filtering
            search_results = await self._search_multimedia_content(
                case_id=case_id,
                query=user_message,
                content_types=question_analysis['target_content_types'],
                limit=15
            )
            
            logger.info(f"Found {len(search_results)} multimedia search results")
            
            # Filter and enhance results based on question type
            relevant_results = await self._filter_multimedia_results(user_message, search_results, question_analysis)
            
            # Build multimedia-aware context
            context_parts = []
            file_sources = []
            
            for result in relevant_results[:5]:
                file_path = result.get('file_path', '')
                file_name = os.path.basename(file_path) if file_path else result.get('id', 'unknown')
                file_type = result.get('file_type', 'document')
                
                # Extract multimedia-specific content
                multimedia_content = await self._extract_multimedia_content_for_context(result, user_message, question_analysis)
                
                if multimedia_content['content']:
                    file_info = {
                        'name': file_name,
                        'path': file_path,
                        'type': file_type,
                        'id': result.get('id', ''),
                        'content_type': multimedia_content['content_type'],
                        'confidence': multimedia_content.get('confidence', 0.5),
                        'relevance_score': multimedia_content.get('relevance_score', 0.5),
                        'timestamp_info': multimedia_content.get('timestamp_info', ''),
                        'analysis_summary': multimedia_content.get('analysis_summary', '')
                    }
                    
                    # Format context based on media type
                    if file_type == 'audio':
                        duration_info = multimedia_content.get('duration', 'Unknown duration')
                        speaker_info = multimedia_content.get('speaker_info', '')
                        context_parts.append(f"🎵 Audio File: {file_name}\n"
                                            f"Duration: {duration_info}\n"
                                            f"{speaker_info}"
                                            f"Transcript: {multimedia_content['content'][:800]}...")
                        
                    elif file_type == 'video':
                        duration_info = multimedia_content.get('duration', 'Unknown duration')
                        visual_info = multimedia_content.get('visual_info', '')
                        context_parts.append(f"🎥 Video File: {file_name}\n"
                                            f"Duration: {duration_info}\n"
                                            f"{visual_info}"
                                            f"Audio Content: {multimedia_content['content'][:800]}...")
                        
                    elif file_type == 'image':
                        image_info = multimedia_content.get('image_info', '')
                        context_parts.append(f"📷 Image File: {file_name}\n"
                                            f"Type: {multimedia_content.get('image_type', 'Unknown')}\n"
                                            f"{image_info}"
                                            f"Text Content: {multimedia_content['content'][:800]}...")
                        
                    else:
                        doc_info = multimedia_content.get('document_info', '')
                        context_parts.append(f"📄 Document: {file_name}\n"
                                            f"{doc_info}"
                                            f"Content: {multimedia_content['content'][:800]}...")
                    
                    file_sources.append(file_info)
            
            # Build multimedia-aware system prompt
            system_prompt = self._build_multimedia_system_prompt(chat_type, question_analysis)
            
            # Create user prompt with multimedia context
            if context_parts:
                context_text = "\n\n".join(context_parts)
                user_prompt = f"""
Question: {user_message}

Available multimedia content from case files:
{context_text}

Please provide a comprehensive answer based on the multimedia content above. Consider:
- Audio/video timestamps and speakers when relevant
- Visual elements and document structure for images
- Cross-reference information across different media types
- Highlight any contradictions or confirmations between sources
- Be specific about which files contain what information

Answer the user's question directly and thoroughly.
"""
            else:
                user_prompt = f"""
Question: {user_message}

I don't have multimedia content that directly answers this question. However, I can help you:
1. Upload additional audio, video, or image files that might contain relevant information
2. Ask questions about specific files you've uploaded
3. Request analysis of particular types of content

What specific multimedia files would help answer your question?
"""
            
            # Call OpenAI with multimedia-optimized settings
            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Lower temperature for factual multimedia analysis
                max_tokens=1200   # More tokens for detailed multimedia responses
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": [f.get('path', '') for f in file_sources],
                "file_sources": file_sources,
                "confidence": self._calculate_multimedia_confidence(file_sources),
                "context_used": len(context_parts) > 0,
                "multimedia_analysis": question_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in multimedia RAG chat: {str(e)}", exc_info=True)
            return {
                "answer": "I apologize, but I encountered an error while analyzing your multimedia content. Please try rephrasing your question or check if your files have been processed correctly.",
                "sources": [],
                "file_sources": [],
                "confidence": 0.0,
                "error": str(e)
            }

    def _analyze_question_intent(self, user_message: str) -> Dict:
        """Analyze what type of multimedia content the user is asking about"""
        message_lower = user_message.lower()
        
        analysis = {
            'target_content_types': ['audio', 'video', 'image', 'document'],
            'temporal_query': False,
            'speaker_query': False,
            'visual_query': False,
            'transcription_query': False,
            'analysis_depth': 'standard',
            'specific_file_query': False,
            'comparison_query': False
        }
        
        # Audio/speech specific queries
        audio_keywords = ['said', 'spoke', 'conversation', 'audio', 'voice', 'transcript', 'recording', 'call', 'phone']
        if any(keyword in message_lower for keyword in audio_keywords):
            analysis['target_content_types'] = ['audio', 'video']  # Video can have audio
            analysis['transcription_query'] = True
        
        # Speaker identification queries
        speaker_keywords = ['who said', 'speaker', 'person speaking', 'voice', 'testimony', 'witness']
        if any(keyword in message_lower for keyword in speaker_keywords):
            analysis['speaker_query'] = True
            analysis['target_content_types'] = ['audio', 'video']
        
        # Visual content queries
        visual_keywords = ['image', 'picture', 'photo', 'document', 'text', 'shows', 'visible', 'written', 'page']
        if any(keyword in message_lower for keyword in visual_keywords):
            analysis['visual_query'] = True
            analysis['target_content_types'] = ['image', 'document']
        
        # Temporal queries
        temporal_keywords = ['when', 'time', 'during', 'at what point', 'timestamp', 'minute', 'second']
        if any(keyword in message_lower for keyword in temporal_keywords):
            analysis['temporal_query'] = True
        
        # Video specific queries
        video_keywords = ['video', 'footage', 'recording', 'clip', 'frame', 'scene']
        if any(keyword in message_lower for keyword in video_keywords):
            analysis['target_content_types'] = ['video']
        
        # Specific file queries
        file_indicators = ['.pdf', '.docx', '.mp3', '.wav', '.mp4', '.jpg', '.png', 'file named', 'document called']
        if any(indicator in message_lower for indicator in file_indicators):
            analysis['specific_file_query'] = True
        
        # Comparison queries
        comparison_keywords = ['compare', 'difference', 'similar', 'contradict', 'consistent', 'versus', 'vs']
        if any(keyword in message_lower for keyword in comparison_keywords):
            analysis['comparison_query'] = True
        
        # Deep analysis requests
        deep_keywords = ['analyze', 'detailed', 'comprehensive', 'breakdown', 'summary', 'explain', 'describe']
        if any(keyword in message_lower for keyword in deep_keywords):
            analysis['analysis_depth'] = 'detailed'
        
        logger.debug(f"Question analysis: {analysis}")
        return analysis

    async def _search_multimedia_content(self, case_id: str, query: str, content_types: List[str], limit: int = 15) -> List[Dict]:
        """Search specifically for multimedia content"""
        try:
            # Use enhanced vector store search with file type filtering
            results = await self.vector_store.search_documents(
                case_id=case_id,
                query=query,
                limit=limit,
                file_types=content_types
            )
            
            logger.debug(f"Vector store returned {len(results)} results for content types: {content_types}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching multimedia content: {e}")
            return []

    async def _filter_multimedia_results(self, query: str, search_results: List[Dict], question_analysis: Dict) -> List[Dict]:
        """Filter and rank multimedia results based on question analysis"""
        if not search_results:
            return []
        
        scored_results = []
        
        for result in search_results:
            relevance_score = await self._calculate_multimedia_relevance(result, query, question_analysis)
            
            if relevance_score > 0.2:  # Minimum threshold
                result['calculated_relevance'] = relevance_score
                scored_results.append(result)
        
        # Sort by relevance score
        scored_results.sort(key=lambda x: x['calculated_relevance'], reverse=True)
        
        logger.debug(f"Filtered to {len(scored_results)} relevant multimedia results")
        return scored_results

    async def _calculate_multimedia_relevance(self, result: Dict, query: str, question_analysis: Dict) -> float:
        """Calculate relevance score for multimedia content"""
        try:
            relevance_score = 0.0
            file_type = result.get('file_type', 'document')
            query_lower = query.lower()
            
            # Base similarity score
            similarity = result.get('similarity_score', 0.5)
            relevance_score += similarity * 0.4
            
            # File type matching bonus
            if file_type in question_analysis['target_content_types']:
                relevance_score += 0.3
            
            # Content-specific relevance
            if file_type == 'audio':
                if question_analysis['transcription_query'] and 'transcript' in result:
                    relevance_score += 0.2
                if question_analysis['speaker_query'] and result.get('speaker_analysis', {}).get('speaker_count', 0) > 1:
                    relevance_score += 0.15
                
                # Check transcript content
                transcript_text = result.get('transcript', {}).get('text', '').lower()
                if transcript_text and any(word in transcript_text for word in query_lower.split()):
                    relevance_score += 0.25
            
            elif file_type == 'video':
                if question_analysis['transcription_query'] and 'audio_analysis' in result:
                    relevance_score += 0.2
                if question_analysis['visual_query'] and result.get('visual_analysis', {}).get('has_text', False):
                    relevance_score += 0.15
                
                # Check video transcript content
                video_transcript = result.get('audio_analysis', {}).get('transcript', {}).get('text', '').lower()
                if video_transcript and any(word in video_transcript for word in query_lower.split()):
                    relevance_score += 0.25
            
            elif file_type == 'image':
                if question_analysis['visual_query'] and 'ocr_results' in result:
                    relevance_score += 0.2
                
                # Check OCR content
                ocr_text = result.get('ocr_results', {}).get('combined_text', '').lower()
                if ocr_text and any(word in ocr_text for word in query_lower.split()):
                    relevance_score += 0.25
                
                # Document type matching
                doc_type = result.get('document_analysis', {}).get('document_type', '')
                if doc_type != 'unknown' and doc_type in query_lower:
                    relevance_score += 0.1
            
            elif file_type == 'document':
                content = result.get('content', '').lower()
                if content and any(word in content for word in query_lower.split()):
                    relevance_score += 0.3
            
            # Temporal query bonus
            if question_analysis['temporal_query']:
                if file_type in ['audio', 'video'] and 'metadata' in result:
                    duration = result['metadata'].get('duration', 0)
                    if duration > 0:
                        relevance_score += 0.1
            
            # Specific file query bonus
            if question_analysis['specific_file_query']:
                filename = result.get('metadata', {}).get('filename', '').lower()
                if any(word in filename for word in query_lower.split()):
                    relevance_score += 0.3
            
            return min(relevance_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating multimedia relevance: {e}")
            return 0.5

    async def _extract_multimedia_content_for_context(self, result: Dict, query: str, question_analysis: Dict) -> Dict:
        """Extract relevant multimedia content based on question analysis"""
        content_info = {
            'content': '',
            'content_type': 'unknown',
            'confidence': 0.5,
            'relevance_score': 0.5,
            'timestamp_info': '',
            'analysis_summary': ''
        }
        
        file_type = result.get('file_type', 'document')
        
        try:
            if file_type == 'audio':
                content_info = await self._extract_audio_content_for_context(result, query, question_analysis)
            elif file_type == 'video':
                content_info = await self._extract_video_content_for_context(result, query, question_analysis)
            elif file_type == 'image':
                content_info = await self._extract_image_content_for_context(result, query, question_analysis)
            else:
                content_info = await self._extract_document_content_for_context(result, query, question_analysis)
                
        except Exception as e:
            logger.warning(f"Error extracting multimedia content: {e}")
        
        return content_info

    async def _extract_audio_content_for_context(self, result: Dict, query: str, question_analysis: Dict) -> Dict:
        """Extract audio-specific content for context"""
        content_info = {
            'content': '',
            'content_type': 'audio_transcript',
            'confidence': 0.5,
            'relevance_score': 0.5,
            'duration': 'Unknown',
            'speaker_info': '',
            'timestamp_info': ''
        }
        
        # Extract transcript
        if 'transcript' in result:
            transcript = result['transcript']
            content_info['content'] = transcript.get('text', '')
            content_info['confidence'] = transcript.get('confidence', 0.5)
            
            # Add formatted transcript with timestamps if available
            formatted_transcript = transcript.get('formatted_transcript', '')
            if formatted_transcript and question_analysis.get('temporal_query'):
                content_info['content'] = formatted_transcript
        
        # Extract duration
        if 'metadata' in result:
            duration = result['metadata'].get('duration', 0)
            if duration:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                content_info['duration'] = f"{minutes}m {seconds}s"
        
        # Add speaker information
        if question_analysis.get('speaker_query') and 'speaker_analysis' in result:
            speaker_info = result['speaker_analysis']
            speaker_count = speaker_info.get('speaker_count', 0)
            if speaker_count > 1:
                content_info['speaker_info'] = f"Speakers detected: {speaker_count}. "
                
                # Add speaker statistics if available
                speaker_stats = speaker_info.get('speaker_statistics', {})
                for speaker_id, stats in speaker_stats.items():
                    participation = stats.get('participation_percentage', 0)
                    if participation > 0:
                        content_info['speaker_info'] += f"{speaker_id}: {participation:.1f}% participation. "
        
        # Add content analysis
        if 'content_analysis' in result:
            content_analysis = result['content_analysis']
            content_type = content_analysis.get('content_type', 'conversation')
            legal_relevance = content_analysis.get('legal_relevance', 'medium')
            
            if content_type != 'conversation':
                content_info['analysis_summary'] = f"Content type: {content_type}. "
            if legal_relevance == 'high':
                content_info['analysis_summary'] += "High legal relevance. "
        
        return content_info

    async def _extract_video_content_for_context(self, result: Dict, query: str, question_analysis: Dict) -> Dict:
        """Extract video-specific content for context"""
        content_info = {
            'content': '',
            'content_type': 'video_content',
            'confidence': 0.5,
            'relevance_score': 0.5,
            'duration': 'Unknown',
            'visual_info': '',
            'timestamp_info': ''
        }
        
        # Extract video audio transcript
        if 'audio_analysis' in result and 'transcript' in result['audio_analysis']:
            transcript = result['audio_analysis']['transcript']
            content_info['content'] = transcript.get('text', '')
            content_info['confidence'] = transcript.get('confidence', 0.5)
            content_info['content_type'] = 'video_transcript'
        
        # Add video metadata
        if 'metadata' in result:
            metadata = result['metadata']
            duration = metadata.get('duration', 0)
            resolution = metadata.get('resolution', 'Unknown')
            
            if duration:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                content_info['duration'] = f"{minutes}m {seconds}s"
            
        # Add visual analysis if available and relevant
        if question_analysis.get('visual_query') and 'visual_analysis' in result:
            visual = result['visual_analysis']
            visual_elements = []
            
            if visual.get('has_faces'):
                visual_elements.append("contains people")
            if visual.get('has_text'):
                visual_elements.append("contains visible text")
            
            scene_changes = visual.get('scene_changes', 0)
            if scene_changes > 0:
                visual_elements.append(f"{scene_changes} scene changes")
                
            if visual_elements:
                content_info['visual_info'] = f"Visual content: {', '.join(visual_elements)}. "
        
        return content_info

    async def _extract_image_content_for_context(self, result: Dict, query: str, question_analysis: Dict) -> Dict:
        """Extract image-specific content for context"""
        content_info = {
            'content': '',
            'content_type': 'image_ocr',
            'confidence': 0.5,
            'relevance_score': 0.5,
            'image_type': 'Unknown',
            'image_info': ''
        }
        
        # Extract OCR text
        if 'ocr_results' in result:
            ocr = result['ocr_results']
            content_info['content'] = ocr.get('combined_text', '')
            content_info['confidence'] = ocr.get('total_confidence', 0) / 100
        
        # Add image metadata
        if 'metadata' in result:
            metadata = result['metadata']
            content_info['image_type'] = metadata.get('image_type', 'unknown')
            resolution = metadata.get('resolution', 'Unknown')
            is_scanned = metadata.get('is_scanned_document', False)
            
            image_info_parts = [f"Resolution: {resolution}"]
            if is_scanned:
                image_info_parts.append("scanned document")
            
            content_info['image_info'] = f"{', '.join(image_info_parts)}. "
        
        # Add document analysis if available
        if 'document_analysis' in result:
            doc_analysis = result['document_analysis']
            doc_type = doc_analysis.get('document_type', 'unknown')
            if doc_type != 'unknown':
                content_info['image_info'] += f"Document type: {doc_type}. "
            
            features = []
            if doc_analysis.get('has_signature_area'):
                features.append("signature area")
            if doc_analysis.get('table_detected'):
                features.append("table data")
            if doc_analysis.get('has_letterhead'):
                features.append("letterhead")
                
            if features:
                content_info['image_info'] += f"Contains: {', '.join(features)}. "
        
        # Add structured data highlights
        if 'structured_data' in result:
            structured = result['structured_data']
            highlights = []
            if structured.get('dates'):
                highlights.append(f"{len(structured['dates'])} dates")
            if structured.get('monetary_amounts'):
                highlights.append(f"{len(structured['monetary_amounts'])} amounts")
            if structured.get('legal_terms'):
                highlights.append(f"{len(structured['legal_terms'])} legal terms")
            if highlights:
                content_info['image_info'] += f"Extracted: {', '.join(highlights)}. "
        
        return content_info

    async def _extract_document_content_for_context(self, result: Dict, query: str, question_analysis: Dict) -> Dict:
        """Extract document-specific content for context"""
        content_info = {
            'content': '',
            'content_type': 'document_text',
            'confidence': 0.9,
            'document_info': ''
        }
        
        # Extract document content
        content = result.get('content', '')
        if content:
            # If it's a long document, try to find the most relevant excerpt
            if len(content) > 1000:
                content_info['content'] = self._find_relevant_excerpt(content, query)
            else:
                content_info['content'] = content
        
        # Add document metadata
        if 'metadata' in result:
            metadata = result['metadata']
            doc_format = metadata.get('format', 'unknown')
            word_count = metadata.get('word_count', 0)
            page_count = metadata.get('page_count', 1)
            
            info_parts = [f"Format: {doc_format.upper()}"]
            if word_count > 0:
                info_parts.append(f"{word_count} words")
            if page_count > 1:
                info_parts.append(f"{page_count} pages")
                
            content_info['document_info'] = f"{', '.join(info_parts)}. "
        
        # Add document structure info
        if 'document_structure' in result:
            structure = result['document_structure']
            doc_type = structure.get('document_type', 'unknown')
            if doc_type != 'unknown':
                content_info['document_info'] += f"Type: {doc_type}. "
                
            features = []
            if structure.get('has_signature_block'):
                features.append("signatures")
            if structure.get('has_letterhead'):
                features.append("letterhead")
            if features:
                content_info['document_info'] += f"Contains: {', '.join(features)}. "
        
        return content_info

    def _find_relevant_excerpt(self, text: str, query: str, context_length: int = 600) -> str:
        """Find the most relevant excerpt from a long document"""
        if not text or not query:
            return text[:context_length] + "..." if len(text) > context_length else text
        
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Find all positions where query words appear
        query_words = query_lower.split()
        positions = []
        
        for word in query_words:
            pos = 0
            while True:
                pos = text_lower.find(word, pos)
                if pos == -1:
                    break
                positions.append(pos)
                pos += 1
        
        if not positions:
            return text[:context_length] + "..." if len(text) > context_length else text
        
        # Find the position with the highest density of query words
        best_pos = min(positions)
        start = max(0, best_pos - context_length // 2)
        end = min(len(text), start + context_length)
        
        excerpt = text[start:end]
        
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(text):
            excerpt = excerpt + "..."
            
        return excerpt

    def _build_multimedia_system_prompt(self, chat_type: str, question_analysis: Dict) -> str:
        """Build system prompt optimized for multimedia content"""
        
        base_prompt = """You are an AI assistant specialized in analyzing multimedia legal discovery content including audio recordings, video files, images, and documents.

When analyzing multimedia content:
- For audio/video: Reference specific speakers, timestamps, and conversation context when available
- For images: Describe visual elements, document structure, and extracted text content
- For documents: Focus on content structure, key information, and document type
- Always cite specific files and provide relevant timestamps or page references when available
- Cross-reference information between different media types when relevant
- Be precise about confidence levels and source reliability
- Use clear, professional language appropriate for legal analysis

IMPORTANT: Only reference files and information that directly relate to the user's question. Focus on providing accurate, relevant information from the available sources."""
        
        # Add specific instructions based on question analysis
        if question_analysis.get('transcription_query'):
            base_prompt += "\n\nFOCUS: Provide accurate transcriptions and quote specific spoken content with timestamps when available."
        
        if question_analysis.get('speaker_query'):
            base_prompt += "\n\nFOCUS: Identify speakers clearly and attribute statements correctly. Note speaker confidence levels."
        
        if question_analysis.get('visual_query'):
            base_prompt += "\n\nFOCUS: Describe visual elements, document layout, text content, and any notable features in images or documents."
        
        if question_analysis.get('temporal_query'):
            base_prompt += "\n\nFOCUS: Provide specific timestamps, time ranges, and chronological context when available."
        
        if question_analysis.get('comparison_query'):
            base_prompt += "\n\nFOCUS: Compare information across different sources and highlight similarities, differences, or contradictions."
        
        if question_analysis.get('analysis_depth') == 'detailed':
            base_prompt += "\n\nProvide comprehensive, detailed analysis with multiple perspectives and thorough examination of the available evidence."
        
        # Add chat type specific instructions
        if chat_type == "privilege_review":
            base_prompt += "\n\nSPECIAL FOCUS: Identify potential attorney-client privileged communications in any media type. Flag any content that might be privileged."
        elif chat_type == "contradiction_analysis":
            base_prompt += "\n\nSPECIAL FOCUS: Look for contradictions, inconsistencies, or conflicts between different sources and media types."
        elif chat_type == "timeline_analysis":
            base_prompt += "\n\nSPECIAL FOCUS: Build chronological timeline from multimedia evidence, noting timestamps and sequence of events."
        
        return base_prompt

    def _calculate_multimedia_confidence(self, file_sources: List[Dict]) -> float:
        """Calculate confidence for multimedia analysis"""
        if not file_sources:
            return 0.0
        
        total_confidence = 0
        total_weight = 0
        
        for source in file_sources:
            # Base confidence from processing
            base_confidence = source.get('confidence', 0.5)
            
            # Weight confidence by media type reliability
            media_type = source.get('type', 'document')
            
            if media_type == 'document':
                weight = 0.9  # Documents usually most reliable
            elif media_type == 'audio':
                # Weight by transcript confidence
                transcript_confidence = source.get('confidence', 0.5)
                weight = 0.8 * transcript_confidence
            elif media_type == 'video':
                # Video audio usually good quality
                weight = 0.8
            elif media_type == 'image':
                # OCR can be variable, weight by confidence
                ocr_confidence = source.get('confidence', 0.5)
                weight = 0.7 * ocr_confidence
            else:
                weight = 0.6
            
            # Factor in relevance score
            relevance = source.get('relevance_score', 0.5)
            
            weighted_confidence = base_confidence * weight * relevance
            total_confidence += weighted_confidence
            total_weight += weight * relevance
        
        return min(total_confidence / total_weight, 1.0) if total_weight > 0 else 0.0

    # Legacy method for backward compatibility
    async def chat_with_case(self, case_id: str, user_message: str, chat_type: str = "general") -> Dict:
        """Legacy method - redirects to multimedia chat"""
        return await self.chat_with_case_multimedia(case_id, user_message, chat_type)

    # New method for searching multimedia content (used by main.py)
    async def search_multimedia_content(self, case_id: str, query: str, content_types: List[str], limit: int = 15) -> List[Dict]:
        """Public method for searching multimedia content"""
        return await self._search_multimedia_content(case_id, query, content_types, limit)