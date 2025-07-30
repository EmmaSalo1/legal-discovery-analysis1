from openai import AsyncOpenAI
from typing import Dict, List, Any, Optional
import logging
import os
import re
from app.config import settings
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        
    async def chat_with_case(self, case_id: str, user_message: str, chat_type: str = "general") -> Dict:
        """Enhanced chat with improved context handling and video analysis"""
        try:
            # Search for relevant documents with higher limit for better filtering
            search_results = await self.vector_store.search_documents(
                case_id=case_id,
                query=user_message,
                limit=20  # Get more candidates to filter from
            )
            
            logger.info(f"Found {len(search_results)} initial search results")
            
            # Filter and rank results by relevance with improved logic
            relevant_results = await self._filter_relevant_results(user_message, search_results)
            
            logger.info(f"Filtered to {len(relevant_results)} relevant results")
            
            # Prepare context from only the most relevant sources
            context_parts = []
            file_sources = []
            
            for result in relevant_results[:8]:  # Use top 8 most relevant
                file_path = result.get('file_path', '')
                file_name = os.path.basename(file_path) if file_path else 'unknown'
                file_type = result.get('file_type', 'document')
                
                # Extract content and check relevance
                content_info = await self._extract_relevant_content(result, user_message)
                
                if content_info['content'] and content_info['relevance_score'] > 0.2:  # Lower threshold for more inclusive results
                    file_info = {
                        'name': file_name,
                        'path': file_path,
                        'type': file_type,
                        'id': result.get('id', ''),
                        'summary': result.get('summary', ''),
                        'content_type': content_info['content_type'],
                        'confidence': content_info['confidence'],
                        'relevance_score': content_info['relevance_score']
                    }
                    
                    # Create more detailed context for different file types
                    context_entry = self._create_context_entry(file_name, file_type, content_info)
                    context_parts.append(context_entry)
                    file_sources.append(file_info)
            
            # Sort file sources by relevance
            file_sources.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Build enhanced system prompt
            system_prompt = self._build_enhanced_system_prompt(chat_type)
            
            # Build user prompt with filtered context
            if context_parts:
                context_text = "\n\n".join(context_parts)
                user_prompt = f"""
Question: {user_message}

Available case evidence and content:
{context_text}

Please provide a comprehensive answer based on the available evidence. If the information directly answers the question, be specific and cite the relevant files. If you're making inferences, clearly indicate that. Focus on the most relevant information for this specific question.
"""
            else:
                user_prompt = f"""
Question: {user_message}

I don't have any relevant content available to answer this specific question about the case. This could mean:
1. The relevant documents haven't been uploaded yet
2. The question requires information not contained in the current files
3. The search terms don't match the content in the documents

Please let me know what specific types of documents or information you're looking for, and I can better guide you on what to upload or how to refine your question.
"""
            
            # Call OpenAI with improved parameters
            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Very low temperature for consistent, factual responses
                max_tokens=800,   # Increased for more detailed responses
                presence_penalty=0.1,  # Slight penalty for repetition
                frequency_penalty=0.1  # Slight penalty for repetitive phrases
            )
            
            answer = response.choices[0].message.content
            
            # Only return sources that were actually relevant and used
            relevant_sources = [f for f in file_sources if f['relevance_score'] > 0.3]
            
            logger.info(f"Returning response with {len(relevant_sources)} sources")
            
            return {
                "answer": answer,
                "sources": [f.get('path', '') for f in relevant_sources],
                "file_sources": relevant_sources,
                "confidence": self._calculate_confidence(relevant_sources),
                "context_used": len(context_parts) > 0,
                "total_files_searched": len(search_results),
                "relevant_files_found": len(relevant_sources)
            }
            
        except Exception as e:
            logger.error(f"Error in RAG chat: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your request. Please ensure your files have been properly uploaded and processed, then try rephrasing your question.",
                "sources": [],
                "file_sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _create_context_entry(self, file_name: str, file_type: str, content_info: Dict) -> str:
        """Create detailed context entry for different file types"""
        content = content_info['content']
        content_type = content_info['content_type']
        confidence = content_info['confidence']
        
        if file_type == 'video':
            if content_type == 'video_transcript':
                return f"VIDEO FILE: {file_name}\nAudio Transcript (Confidence: {confidence:.1%}):\n{content}"
            elif content_type == 'visual_content':
                return f"VIDEO FILE: {file_name}\nVisual Analysis:\n{content}"
            else:
                return f"VIDEO FILE: {file_name}\nContent:\n{content}"
                
        elif file_type == 'audio':
            return f"AUDIO FILE: {file_name}\nTranscript (Confidence: {confidence:.1%}):\n{content}"
            
        elif file_type == 'image':
            return f"IMAGE FILE: {file_name}\nOCR Text (Confidence: {confidence:.1%}):\n{content}"
            
        elif file_type == 'document':
            return f"DOCUMENT: {file_name}\nContent:\n{content}"
        
        else:
            return f"FILE: {file_name}\nContent:\n{content}"
    
    async def _filter_relevant_results(self, query: str, search_results: List[Dict]) -> List[Dict]:
        """Enhanced filtering with better relevance scoring"""
        if not search_results:
            return []
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        # Enhanced stop words list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'what', 'how', 'when', 'where', 'why', 'who', 'is', 'are', 'was', 'were', 'do', 'does', 
            'did', 'have', 'has', 'had', 'can', 'could', 'should', 'would', 'will', 'shall',
            'tell', 'me', 'about', 'show', 'find', 'get', 'give', 'please', 'this', 'that'
        }
        query_words = query_words - stop_words
        
        scored_results = []
        
        for result in search_results:
            relevance_score = await self._calculate_content_relevance(result, query_words, query_lower)
            
            if relevance_score > 0.05:  # Very low threshold for initial filtering
                result['calculated_relevance'] = relevance_score
                scored_results.append(result)
        
        # Sort by relevance score
        scored_results.sort(key=lambda x: x['calculated_relevance'], reverse=True)
        
        return scored_results
    
    async def _calculate_content_relevance(self, result: Dict, query_words: set, query_lower: str) -> float:
        """Enhanced relevance calculation with support for all file types"""
        relevance_score = 0.0
        
        # Get all searchable text from the document based on file type
        searchable_texts = []
        
        # Handle different file types
        file_type = result.get('file_type', 'document')
        
        if file_type == 'audio' and 'transcript' in result:
            transcript_text = result['transcript'].get('text', '')
            if transcript_text:
                searchable_texts.append(('transcript', transcript_text, 1.0))  # High weight for transcripts
        
        elif file_type == 'video':
            # Check for audio analysis
            if 'audio_analysis' in result and 'transcript' in result['audio_analysis']:
                transcript_text = result['audio_analysis']['transcript'].get('text', '')
                if transcript_text:
                    searchable_texts.append(('video_transcript', transcript_text, 1.0))
            
            # Check for visual analysis
            if 'visual_analysis' in result and 'visual_summary' in result['visual_analysis']:
                visual_summary = result['visual_analysis']['visual_summary']
                if isinstance(visual_summary, list):
                    visual_text = ' '.join([item.get('description', '') for item in visual_summary])
                    if visual_text:
                        searchable_texts.append(('visual_content', visual_text, 0.5))
        
        elif file_type == 'image' and 'ocr_results' in result:
            ocr_text = result['ocr_results'].get('combined_text', '')
            if ocr_text:
                searchable_texts.append(('ocr_text', ocr_text, 0.8))
        
        elif file_type == 'document' and 'content' in result:
            document_content = result.get('content', '')
            if document_content:
                searchable_texts.append(('document_content', document_content, 1.0))
        
        # Also check summary and filename
        if 'summary' in result and result['summary']:
            searchable_texts.append(('summary', result['summary'], 0.7))
        
        file_path = result.get('file_path', '')
        if file_path:
            filename = os.path.basename(file_path)
            searchable_texts.append(('filename', filename, 0.6))
        
        # Calculate relevance based on text matches
        for content_type, text, weight in searchable_texts:
            if not text:
                continue
                
            text_lower = text.lower()
            
            # Exact phrase match (highest score)
            if query_lower in text_lower:
                relevance_score += 3.0 * weight
            
            # Individual word matches
            text_words = set(re.findall(r'\w+', text_lower))
            word_matches = query_words.intersection(text_words)
            
            if word_matches:
                # Score based on percentage of query words found and frequency
                word_match_ratio = len(word_matches) / len(query_words) if query_words else 0
                relevance_score += word_match_ratio * 2.0 * weight
                
                # Bonus for multiple occurrences of matched words
                for word in word_matches:
                    occurrence_count = text_lower.count(word)
                    if occurrence_count > 1:
                        relevance_score += min(occurrence_count * 0.1, 0.5) * weight  # Cap bonus
        
        # Bonus for file type relevance
        if any(word in query_lower for word in ['audio', 'transcript', 'recording', 'voice', 'speech']):
            if file_type == 'audio':
                relevance_score += 0.5
            elif file_type == 'video' and 'audio_analysis' in result:
                relevance_score += 0.3
        
        if any(word in query_lower for word in ['video', 'recording', 'footage', 'visual']):
            if file_type == 'video':
                relevance_score += 0.5
        
        if any(word in query_lower for word in ['image', 'photo', 'picture', 'scan', 'document']):
            if file_type in ['image', 'document']:
                relevance_score += 0.3
        
        return min(relevance_score, 5.0)  # Cap at 5.0
    
    async def _extract_relevant_content(self, result: Dict, query: str) -> Dict:
        """Enhanced content extraction for all file types"""
        file_type = result.get('file_type', 'document')
        query_lower = query.lower()
        
        content_info = {
            'content': '',
            'content_type': 'unknown',
            'confidence': 0.0,
            'relevance_score': 0.0
        }
        
        full_text = ""
        
        # Extract content based on file type
        if file_type == 'audio' and 'transcript' in result:
            full_text = result['transcript'].get('text', '')
            content_info['content_type'] = 'transcript'
            content_info['confidence'] = result['transcript'].get('confidence', 0)
            
        elif file_type == 'video':
            # Prioritize transcript content
            if 'audio_analysis' in result and 'transcript' in result['audio_analysis']:
                full_text = result['audio_analysis']['transcript'].get('text', '')
                content_info['content_type'] = 'video_transcript'
                content_info['confidence'] = result['audio_analysis']['transcript'].get('confidence', 0)
            
            # If no transcript, use visual analysis
            elif 'visual_analysis' in result and 'visual_summary' in result['visual_analysis']:
                visual_summary = result['visual_analysis']['visual_summary']
                if isinstance(visual_summary, list):
                    full_text = '; '.join([
                        f"{item.get('timestamp_formatted', '')}: {item.get('description', '')}"
                        for item in visual_summary if item.get('description')
                    ])
                    content_info['content_type'] = 'visual_content'
                    content_info['confidence'] = 0.7  # Moderate confidence for visual analysis
            
        elif file_type == 'image' and 'ocr_results' in result:
            full_text = result['ocr_results'].get('combined_text', '')
            content_info['content_type'] = 'ocr_text'
            content_info['confidence'] = result['ocr_results'].get('total_confidence', 0) / 100
            
        elif file_type == 'document' and 'content' in result:
            full_text = result.get('content', '')
            content_info['content_type'] = 'document_text'
            content_info['confidence'] = 0.95  # High confidence for extracted text
        
        if not full_text:
            return content_info
        
        # Find the most relevant excerpt
        relevant_excerpt = self._find_relevant_excerpt(full_text, query_lower)
        content_info['content'] = relevant_excerpt
        
        # Calculate relevance score for this specific content
        content_info['relevance_score'] = self._score_content_relevance(relevant_excerpt, query_lower)
        
        return content_info
    
    def _find_relevant_excerpt(self, text: str, query_lower: str, max_length: int = 600) -> str:
        """Find the most relevant excerpt with improved context window"""
        if not text or not query_lower:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        text_lower = text.lower()
        
        # If query phrase is found, extract context around it
        query_pos = text_lower.find(query_lower)
        if query_pos != -1:
            # Extract larger context around the query
            context_size = max_length // 2
            start = max(0, query_pos - context_size)
            end = min(len(text), query_pos + len(query_lower) + context_size)
            
            excerpt = text[start:end]
            
            # Try to break at sentence boundaries
            if start > 0:
                # Find first complete sentence
                first_period = excerpt.find('. ')
                if first_period > 0 and first_period < 100:
                    excerpt = excerpt[first_period + 2:]
                else:
                    excerpt = "..." + excerpt
                    
            if end < len(text):
                # Find last complete sentence
                last_period = excerpt.rfind('. ')
                if last_period > len(excerpt) - 100:
                    excerpt = excerpt[:last_period + 1]
                else:
                    excerpt = excerpt + "..."
            
            return excerpt
        
        # If no exact match, look for individual words and find best section
        query_words = re.findall(r'\w+', query_lower)
        best_excerpt = ""
        best_score = 0
        
        # Split text into overlapping chunks
        chunk_size = max_length
        overlap = chunk_size // 4
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunk_lower = chunk.lower()
            
            # Score this chunk based on query word matches
            score = sum(1 for word in query_words if word in chunk_lower)
            
            if score > best_score:
                best_score = score
                best_excerpt = chunk
        
        if best_excerpt:
            return best_excerpt
        
        # Fallback to beginning of text
        return text[:max_length] + "..." if len(text) > max_length else text
    
    def _score_content_relevance(self, content: str, query_lower: str) -> float:
        """Enhanced content relevance scoring"""
        if not content or not query_lower:
            return 0.0
        
        content_lower = content.lower()
        
        # Exact phrase match
        if query_lower in content_lower:
            return 1.0
        
        # Individual word matches with position weighting
        query_words = re.findall(r'\w+', query_lower)
        content_words = re.findall(r'\w+', content_lower)
        
        if not query_words:
            return 0.0
        
        matches = 0
        total_score = 0.0
        
        for word in query_words:
            if word in content_words:
                matches += 1
                # Count occurrences for frequency bonus
                word_count = content_lower.count(word)
                total_score += min(word_count * 0.1, 0.5)  # Cap frequency bonus
        
        base_score = matches / len(query_words)
        return min(base_score + total_score, 1.0)
    
    def _build_enhanced_system_prompt(self, chat_type: str) -> str:
        """Enhanced system prompt for better responses"""
        base_prompt = """You are an AI assistant specialized in legal discovery analysis. You help lawyers and legal professionals analyze case documents, audio/video evidence, and other legal materials.

IMPORTANT GUIDELINES:
- Only reference information that directly relates to the user's question
- Be specific and cite relevant files when possible
- If making inferences, clearly indicate that
- Focus on factual information from the provided evidence
- For video files, distinguish between audio transcript content and visual analysis
- Maintain professional legal analysis standards
- If information is missing, suggest what documents might be needed

Your responses should be:
- Comprehensive yet focused on the question asked
- Professional and legally appropriate
- Clear about the source and type of information (transcript, document, OCR, etc.)
- Honest about limitations in the available evidence"""
        
        if chat_type == "privilege_review":
            base_prompt += "\n\nFOCUS: Identify attorney-client privileged communications, work product, or other potentially privileged materials. Flag any privilege concerns clearly."
        elif chat_type == "contradiction_analysis":
            base_prompt += "\n\nFOCUS: Look for contradictions, inconsistencies, or conflicting statements in the evidence. Highlight discrepancies between different sources."
        elif chat_type == "timeline_analysis":
            base_prompt += "\n\nFOCUS: Analyze temporal relationships and chronological sequences of events. Help construct accurate timelines."
        elif chat_type == "evidence_analysis":
            base_prompt += "\n\nFOCUS: Identify key evidence, assess its relevance and potential impact, and note any evidentiary issues."
        
        return base_prompt
    
    def _calculate_confidence(self, file_sources: List[Dict]) -> float:
        """Calculate overall confidence based on source quality and relevance"""
        if not file_sources:
            return 0.0
        
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for source in file_sources:
            source_confidence = source.get('confidence', 0.5)
            relevance_score = source.get('relevance_score', 0.5)
            
            # Weight confidence by relevance
            weight = relevance_score
            total_weighted_confidence += source_confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        base_confidence = total_weighted_confidence / total_weight
        
        # Adjust based on number of sources (more sources = higher confidence)
        source_count_factor = min(len(file_sources) / 3.0, 1.0)  # Cap at 3 sources
        
        final_confidence = base_confidence * (0.7 + 0.3 * source_count_factor)
        
        return min(final_confidence, 1.0)