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
        """Enhanced chat with improved source relevance filtering"""
        try:
            # Search for relevant documents
            search_results = await self.vector_store.search_documents(
                case_id=case_id,
                query=user_message,
                limit=15  # Get more candidates to filter from
            )
            
            # Filter and rank results by relevance
            relevant_results = await self._filter_relevant_results(user_message, search_results)
            
            # Prepare context from only the most relevant sources
            context_parts = []
            file_sources = []
            
            for result in relevant_results[:5]:  # Use only top 5 most relevant
                file_path = result.get('file_path', '')
                file_name = os.path.basename(file_path) if file_path else 'unknown'
                file_type = result.get('file_type', 'document')
                
                # Extract content and check relevance
                content_info = await self._extract_relevant_content(result, user_message)
                
                if content_info['content'] and content_info['relevance_score'] > 0.3:  # Minimum relevance threshold
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
                    
                    context_parts.append(f"File: {file_name}\nContent: {content_info['content']}")
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

Relevant case files with content:
{context_text}

Please provide a clear, concise answer based ONLY on the information that directly relates to the question. If some files don't contain relevant information for this specific question, don't mention them in your response.
"""
            else:
                user_prompt = f"""
Question: {user_message}

I don't have any relevant content available to answer this specific question. Please let the user know what files are available and suggest they may need to upload more specific documents or ask a different question.
"""
            
            # Call OpenAI
            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Lower temperature for more focused responses
                max_tokens=600
            )
            
            answer = response.choices[0].message.content
            
            # Only return sources that were actually relevant
            relevant_sources = [f for f in file_sources if f['relevance_score'] > 0.5]
            
            return {
                "answer": answer,
                "sources": [f.get('path', '') for f in relevant_sources],
                "file_sources": relevant_sources,
                "confidence": self._calculate_confidence(relevant_sources),
                "context_used": len(context_parts) > 0
            }
            
        except Exception as e:
            logger.error(f"Error in RAG chat: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your request. Please try again.",
                "sources": [],
                "file_sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _filter_relevant_results(self, query: str, search_results: List[Dict]) -> List[Dict]:
        """Filter search results by relevance to the specific query"""
        if not search_results:
            return []
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        # Remove common stop words that don't help with relevance
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'how', 'when', 'where', 'why', 'who', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'have', 'has', 'had', 'can', 'could', 'should', 'would', 'will'}
        query_words = query_words - stop_words
        
        scored_results = []
        
        for result in search_results:
            relevance_score = await self._calculate_content_relevance(result, query_words, query_lower)
            
            if relevance_score > 0.1:  # Minimum threshold
                result['calculated_relevance'] = relevance_score
                scored_results.append(result)
        
        # Sort by relevance score
        scored_results.sort(key=lambda x: x['calculated_relevance'], reverse=True)
        
        return scored_results
    
    async def _calculate_content_relevance(self, result: Dict, query_words: set, query_lower: str) -> float:
        """Calculate how relevant a document is to the specific query"""
        relevance_score = 0.0
        
        # Get all searchable text from the document
        searchable_texts = []
        
        # Extract text based on document type
        if result.get('file_type') == 'audio' and 'transcript' in result:
            searchable_texts.append(result['transcript'].get('text', ''))
        
        elif result.get('file_type') == 'video' and 'audio_analysis' in result:
            if 'transcript' in result['audio_analysis']:
                searchable_texts.append(result['audio_analysis']['transcript'].get('text', ''))
        
        elif result.get('file_type') == 'image' and 'ocr_results' in result:
            searchable_texts.append(result['ocr_results'].get('combined_text', ''))
        
        elif result.get('file_type') == 'document' and 'content' in result:
            searchable_texts.append(result.get('content', ''))
        
        # Also check summary and filename
        if 'summary' in result:
            searchable_texts.append(result['summary'])
        
        file_path = result.get('file_path', '')
        if file_path:
            searchable_texts.append(os.path.basename(file_path))
        
        # Calculate relevance based on word matches and context
        for text in searchable_texts:
            if not text:
                continue
                
            text_lower = text.lower()
            
            # Exact phrase match (high score)
            if query_lower in text_lower:
                relevance_score += 2.0
            
            # Individual word matches
            text_words = set(re.findall(r'\w+', text_lower))
            word_matches = query_words.intersection(text_words)
            
            if word_matches:
                # Score based on percentage of query words found
                word_match_ratio = len(word_matches) / len(query_words) if query_words else 0
                relevance_score += word_match_ratio * 1.0
                
                # Bonus for multiple word matches in close proximity
                for word in word_matches:
                    if text_lower.count(word) > 1:
                        relevance_score += 0.1
        
        # Filename relevance bonus
        filename = os.path.basename(result.get('file_path', ''))
        filename_lower = filename.lower()
        for word in query_words:
            if word in filename_lower:
                relevance_score += 0.3
        
        return min(relevance_score, 3.0)  # Cap at 3.0
    
    async def _extract_relevant_content(self, result: Dict, query: str) -> Dict:
        """Extract the most relevant content from a document for the specific query"""
        file_type = result.get('file_type', 'document')
        query_lower = query.lower()
        
        content_info = {
            'content': '',
            'content_type': 'unknown',
            'confidence': 0.0,
            'relevance_score': 0.0
        }
        
        # Extract content based on file type
        if file_type == 'audio' and 'transcript' in result:
            full_text = result['transcript'].get('text', '')
            content_info['content_type'] = 'transcript'
            content_info['confidence'] = result['transcript'].get('confidence', 0)
            
        elif file_type == 'video' and 'audio_analysis' in result:
            if 'transcript' in result['audio_analysis']:
                full_text = result['audio_analysis']['transcript'].get('text', '')
                content_info['content_type'] = 'video_transcript'
                content_info['confidence'] = result['audio_analysis']['transcript'].get('confidence', 0)
            
        elif file_type == 'image' and 'ocr_results' in result:
            full_text = result['ocr_results'].get('combined_text', '')
            content_info['content_type'] = 'ocr_text'
            content_info['confidence'] = result['ocr_results'].get('total_confidence', 0) / 100
            
        elif file_type == 'document' and 'content' in result:
            full_text = result.get('content', '')
            content_info['content_type'] = 'document_text'
            content_info['confidence'] = 0.9  # High confidence for extracted text
        
        else:
            return content_info
        
        if not full_text:
            return content_info
        
        # Find the most relevant excerpt
        relevant_excerpt = self._find_relevant_excerpt(full_text, query_lower)
        content_info['content'] = relevant_excerpt
        
        # Calculate relevance score for this specific content
        content_info['relevance_score'] = self._score_content_relevance(relevant_excerpt, query_lower)
        
        return content_info
    
    def _find_relevant_excerpt(self, text: str, query_lower: str) -> str:
        """Find the most relevant excerpt from the text"""
        if not text or not query_lower:
            return text[:400] + "..." if len(text) > 400 else text
        
        # If query is found in text, extract around it
        text_lower = text.lower()
        query_pos = text_lower.find(query_lower)
        
        if query_pos != -1:
            # Extract context around the query
            start = max(0, query_pos - 200)
            end = min(len(text), query_pos + len(query_lower) + 200)
            excerpt = text[start:end]
            
            # Add ellipsis if truncated
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(text):
                excerpt = excerpt + "..."
                
            return excerpt
        
        # If no exact match, look for individual words
        query_words = re.findall(r'\w+', query_lower)
        best_excerpt = ""
        best_score = 0
        
        # Split text into chunks and score each
        words = text.split()
        chunk_size = 100
        
        for i in range(0, len(words), chunk_size // 2):  # Overlapping chunks
            chunk = " ".join(words[i:i + chunk_size])
            chunk_lower = chunk.lower()
            
            score = sum(1 for word in query_words if word in chunk_lower)
            
            if score > best_score:
                best_score = score
                best_excerpt = chunk
        
        return best_excerpt if best_excerpt else text[:400] + "..." if len(text) > 400 else text
    
    def _score_content_relevance(self, content: str, query_lower: str) -> float:
        """Score how relevant the content is to the query"""
        if not content or not query_lower:
            return 0.0
        
        content_lower = content.lower()
        
        # Exact phrase match
        if query_lower in content_lower:
            return 1.0
        
        # Individual word matches
        query_words = re.findall(r'\w+', query_lower)
        content_words = re.findall(r'\w+', content_lower)
        
        if not query_words:
            return 0.0
        
        matches = sum(1 for word in query_words if word in content_words)
        return matches / len(query_words)
    
    def _build_enhanced_system_prompt(self, chat_type: str) -> str:
        """Build enhanced system prompt for better responses"""
        base_prompt = """You are an AI assistant specialized in legal discovery analysis. 

IMPORTANT: Only reference files and information that directly relate to the user's question. Do not mention files that don't contain relevant information for the specific question asked.

Guidelines:
- Provide clear, concise answers based only on relevant content
- Focus on the actual substance that answers the question
- If you don't have relevant information, say so clearly
- Avoid mentioning files that don't relate to the specific question
- Be direct and helpful in your analysis"""
        
        if chat_type == "privilege_review":
            base_prompt += "\n\nFocus: Identify attorney-client privileged communications or work product."
        elif chat_type == "contradiction_analysis":
            base_prompt += "\n\nFocus: Look for contradictions or inconsistencies in the evidence."
        elif chat_type == "timeline_analysis":
            base_prompt += "\n\nFocus: Analyze temporal relationships and chronological sequence."
        
        return base_prompt
    
    def _calculate_confidence(self, file_sources: List[Dict]) -> float:
        """Calculate confidence based on relevant sources only"""
        if not file_sources:
            return 0.0
        
        total_confidence = sum(
            source.get('confidence', 0.5) * source.get('relevance_score', 0.5) 
            for source in file_sources
        )
        
        return min(total_confidence / len(file_sources), 1.0)