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
        """Enhanced chat with improved source relevance filtering - FIXED VERSION"""
        try:
            logger.info(f"Processing chat request for case {case_id}: {user_message}")
            
            # Search for relevant documents with LOWER threshold
            search_results = await self.vector_store.search_documents(
                case_id=case_id,
                query=user_message,
                limit=20  # Get more candidates
            )
            
            logger.info(f"Vector search returned {len(search_results)} results")
            
            # If no results, try a broader search
            if not search_results:
                logger.warning("No results from initial search, trying broader search")
                # Try searching for individual words
                words = user_message.split()
                for word in words:
                    if len(word) > 3:  # Only search meaningful words
                        broader_results = await self.vector_store.search_documents(
                            case_id=case_id,
                            query=word,
                            limit=10
                        )
                        search_results.extend(broader_results)
                        if len(search_results) >= 5:
                            break
            
            # Filter and rank results by relevance with MUCH lower threshold
            relevant_results = await self._filter_relevant_results(user_message, search_results)
            
            logger.info(f"After relevance filtering: {len(relevant_results)} results")
            
            # Prepare context from the most relevant sources
            context_parts = []
            file_sources = []
            
            # Use MORE results and LOWER threshold
            for result in relevant_results[:10]:  # Use top 10 instead of 5
                file_path = result.get('file_path', '')
                file_name = os.path.basename(file_path) if file_path else 'unknown'
                file_type = result.get('file_type', 'document')
                
                # Extract content with MUCH lower relevance threshold
                content_info = await self._extract_relevant_content(result, user_message)
                
                # EXTREMELY low threshold - almost everything should pass now
                if content_info['content'] and content_info['relevance_score'] > 0.01:  # Changed from 0.1 to 0.01
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
            
            logger.info(f"Using {len(context_parts)} files for context")
            
            # Sort file sources by relevance
            file_sources.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Build enhanced system prompt
            system_prompt = self._build_enhanced_system_prompt(chat_type)
            
            # Build user prompt with context - ALWAYS provide available files info
            if context_parts:
                context_text = "\n\n".join(context_parts)
                user_prompt = f"""
Question: {user_message}

Available case files with relevant content:
{context_text}

Please provide a helpful answer based on the available information. If the specific information requested isn't available, explain what files are available and suggest alternative questions.
"""
            else:
                # IMPROVED fallback response with actual file list
                try:
                    # Get list of available files
                    all_results = await self.vector_store.search_documents(case_id, "", limit=50)
                    file_list = []
                    seen_files = set()
                    
                    for result in all_results:
                        filename = result.get('metadata', {}).get('filename') or os.path.basename(result.get('file_path', ''))
                        if filename and filename not in seen_files:
                            file_list.append(filename)
                            seen_files.add(filename)
                    
                    if file_list:
                        files_text = "\n".join([f"- {f}" for f in file_list[:20]])  # Show first 20
                        user_prompt = f"""
Question: {user_message}

I don't have specific content that directly answers your question, but I have access to these case files:

{files_text}

Please let the user know what files are available and suggest they ask more specific questions about these files, or ask me to summarize the content of specific files.
"""
                    else:
                        user_prompt = f"""
Question: {user_message}

I don't have access to any case files for this case. Please make sure files have been uploaded and processed correctly.
"""
                        
                except Exception as e:
                    logger.error(f"Error getting file list: {e}")
                    user_prompt = f"""
Question: {user_message}

I'm having trouble accessing the case files. Please check that files have been uploaded and processed correctly.
"""
            
            # Call OpenAI
            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=600
            )
            
            answer = response.choices[0].message.content
            
            # Return sources that were actually used (extremely low threshold)
            relevant_sources = [f for f in file_sources if f['relevance_score'] > 0.01]  # Changed from 0.1 to 0.01
            
            return {
                "answer": answer,
                "sources": [f.get('path', '') for f in relevant_sources],
                "file_sources": relevant_sources,
                "confidence": self._calculate_confidence(relevant_sources),
                "context_used": len(context_parts) > 0,
                "debug_info": {
                    "total_search_results": len(search_results),
                    "relevant_results": len(relevant_results),
                    "context_parts": len(context_parts)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in RAG chat: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your request. Please try again or check the system logs for more details.",
                "sources": [],
                "file_sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _filter_relevant_results(self, query: str, search_results: List[Dict]) -> List[Dict]:
        """Filter search results by relevance - EXTREMELY permissive"""
        if not search_results:
            return []
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        # Remove fewer stop words to catch more connections
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but'}
        query_words = query_words - stop_words
        
        # Add semantic expansion for legal terms
        expanded_query_words = self._expand_legal_query(query_words)
        
        scored_results = []
        
        for result in search_results:
            relevance_score = await self._calculate_content_relevance(result, expanded_query_words, query_lower)
            
            # EXTREMELY low threshold - almost everything passes
            if relevance_score > 0.01:  # Reduced from 0.05 to 0.01
                result['calculated_relevance'] = relevance_score
                scored_results.append(result)
        
        # If still no results, include everything with any score
        if not scored_results and search_results:
            for result in search_results:
                result['calculated_relevance'] = 0.1  # Give everything a base score
                scored_results.append(result)
        
        # Sort by relevance score
        scored_results.sort(key=lambda x: x['calculated_relevance'], reverse=True)
        
        return scored_results
    
    def _expand_legal_query(self, query_words: set) -> set:
        """Expand query with legal synonyms and related terms"""
        expanded_words = set(query_words)
        
        # Legal term expansions
        legal_expansions = {
            'witness': ['interview', 'statement', 'testimony', 'deposition', 'account'],
            'tip': ['report', 'information', 'lead', 'tip', 'cjis', 'investigative'],
            'tips': ['reports', 'information', 'leads', 'tips', 'cjis', 'investigative'],
            'information': ['data', 'details', 'facts', 'evidence', 'content'],
            'phone': ['cell', 'mobile', 'telephone', 'call', 'contact', 'lg', 'iphone'],
            'report': ['document', 'file', 'analysis', 'summary', 'study'],
            'interview': ['conversation', 'discussion', 'questioning', 'statement'],
            'evidence': ['proof', 'documentation', 'material', 'exhibit'],
            'investigation': ['inquiry', 'probe', 'examination', 'review'],
            'suspect': ['person', 'individual', 'subject', 'defendant'],
            'alibi': ['whereabouts', 'location', 'timeline', 'schedule']
        }
        
        for word in list(query_words):
            if word in legal_expansions:
                expanded_words.update(legal_expansions[word])
        
        return expanded_words
    
    async def _calculate_content_relevance(self, result: Dict, query_words: set, query_lower: str) -> float:
        """Calculate how relevant a document is - EXTREMELY permissive scoring"""
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
        
        # Check metadata filename
        if 'metadata' in result and result['metadata'].get('filename'):
            searchable_texts.append(result['metadata']['filename'])
        
        # Calculate relevance based on word matches and context
        for text in searchable_texts:
            if not text:
                continue
                
            text_lower = text.lower()
            
            # Exact phrase match (high score)
            if query_lower in text_lower:
                relevance_score += 3.0  # Increased from 2.0
            
            # Individual word matches - VERY generous scoring
            text_words = set(re.findall(r'\w+', text_lower))
            word_matches = query_words.intersection(text_words)
            
            if word_matches:
                # Much more generous scoring for word matches
                word_match_ratio = len(word_matches) / max(len(query_words), 1)
                relevance_score += word_match_ratio * 2.0  # Increased from 1.5
                
                # Additional bonus for multiple matches
                if len(word_matches) > 1:
                    relevance_score += 1.0  # Increased from 0.5
                
                # Bonus for each matched word
                relevance_score += len(word_matches) * 0.3
            
            # Partial word matches (for related terms)
            for query_word in query_words:
                if len(query_word) > 3:  # Only for meaningful words
                    for text_word in text_words:
                        if query_word in text_word or text_word in query_word:
                            relevance_score += 0.2
        
        # Filename relevance bonus - VERY generous
        filename = os.path.basename(result.get('file_path', ''))
        filename_lower = filename.lower()
        for word in query_words:
            if word in filename_lower:
                relevance_score += 1.0  # Increased from 0.5
            # Partial filename matches
            elif len(word) > 3:
                for filename_part in filename_lower.split():
                    if word in filename_part or filename_part in word:
                        relevance_score += 0.5
        
        # Special bonus for legal document patterns in filename
        legal_patterns = ['tip', 'report', 'interview', 'cjis', 'statement', 'witness', 'investigation']
        for pattern in legal_patterns:
            if pattern in filename_lower:
                relevance_score += 0.5
        
        return min(relevance_score, 10.0)  # Increased cap from 5.0 to 10.0
    
    async def _extract_relevant_content(self, result: Dict, query: str) -> Dict:
        """Extract the most relevant content - MORE permissive"""
        file_type = result.get('file_type', 'document')
        query_lower = query.lower()
        
        content_info = {
            'content': '',
            'content_type': 'unknown',
            'confidence': 0.0,
            'relevance_score': 0.0
        }
        
        # Extract content based on file type
        full_text = ''
        
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
            content_info['confidence'] = 0.9
        
        if not full_text:
            # Fallback to any available text
            if 'summary' in result:
                full_text = result['summary']
                content_info['content_type'] = 'summary'
                content_info['confidence'] = 0.7
        
        if full_text:
            # Get relevant excerpt - MORE generous
            relevant_excerpt = self._find_relevant_excerpt(full_text, query_lower)
            content_info['content'] = relevant_excerpt
            
            # More generous relevance scoring
            content_info['relevance_score'] = self._score_content_relevance(relevant_excerpt, query_lower)
        
        return content_info
    
    def _find_relevant_excerpt(self, text: str, query_lower: str) -> str:
        """Find relevant excerpt - return MORE content"""
        if not text or not query_lower:
            return text[:800] + "..." if len(text) > 800 else text  # Increased from 400
        
        # If query is found in text, extract around it
        text_lower = text.lower()
        query_pos = text_lower.find(query_lower)
        
        if query_pos != -1:
            # Extract MORE context around the query
            start = max(0, query_pos - 300)  # Increased from 200
            end = min(len(text), query_pos + len(query_lower) + 300)  # Increased from 200
            excerpt = text[start:end]
            
            # Add ellipsis if truncated
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(text):
                excerpt = excerpt + "..."
                
            return excerpt
        
        # If no exact match, look for individual words and return MORE content
        query_words = re.findall(r'\w+', query_lower)
        best_excerpt = ""
        best_score = 0
        
        # Split text into larger chunks
        words = text.split()
        chunk_size = 150  # Increased from 100
        
        for i in range(0, len(words), chunk_size // 2):  # Overlapping chunks
            chunk = " ".join(words[i:i + chunk_size])
            chunk_lower = chunk.lower()
            
            score = sum(1 for word in query_words if word in chunk_lower)
            
            if score > best_score:
                best_score = score
                best_excerpt = chunk
        
        return best_excerpt if best_excerpt else text[:800] + "..." if len(text) > 800 else text
    
    def _score_content_relevance(self, content: str, query_lower: str) -> float:
        """Score content relevance - MORE generous"""
        if not content or not query_lower:
            return 0.0
        
        content_lower = content.lower()
        
        # Exact phrase match
        if query_lower in content_lower:
            return 1.0
        
        # Individual word matches - more generous
        query_words = re.findall(r'\w+', query_lower)
        content_words = re.findall(r'\w+', content_lower)
        
        if not query_words:
            return 0.0
        
        matches = sum(1 for word in query_words if word in content_words)
        base_score = matches / len(query_words)
        
        # Bonus for multiple matches
        if matches > 1:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    def _build_enhanced_system_prompt(self, chat_type: str) -> str:
        """Build enhanced system prompt for better responses"""
        base_prompt = """You are an AI assistant specialized in legal discovery analysis. 

IMPORTANT: Always provide helpful responses based on available information. If you don't have specific information to answer a question, explain what files are available and suggest alternative approaches.

Guidelines:
- Provide clear, useful answers based on available content
- If specific information isn't available, explain what is available
- Be direct and helpful in your analysis
- Suggest specific follow-up questions when appropriate"""
        
        if chat_type == "privilege_review":
            base_prompt += "\n\nFocus: Identify attorney-client privileged communications or work product."
        elif chat_type == "contradiction_analysis":
            base_prompt += "\n\nFocus: Look for contradictions or inconsistencies in the evidence."
        elif chat_type == "timeline_analysis":
            base_prompt += "\n\nFocus: Analyze temporal relationships and chronological sequence."
        
        return base_prompt
    
    def _calculate_confidence(self, file_sources: List[Dict]) -> float:
        """Calculate confidence - more generous"""
        if not file_sources:
            return 0.0
        
        total_confidence = sum(
            source.get('confidence', 0.5) * source.get('relevance_score', 0.5) 
            for source in file_sources
        )
        
        avg_confidence = total_confidence / len(file_sources)
        
        # Boost confidence if we have multiple sources
        if len(file_sources) > 1:
            avg_confidence = min(avg_confidence * 1.2, 1.0)
        
        return avg_confidence