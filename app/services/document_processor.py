from datetime import datetime
import os
import mimetypes
import logging
from typing import Dict, Any, List
from app.services.audio_processor import AudioProcessor
from app.services.video_processor import VideoProcessor
from app.services.image_processor import ImageProcessor
from app.config import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        self.image_processor = ImageProcessor()
        
    async def process_document(self, file_path: str, case_id: str) -> Dict[str, Any]:
        """FIXED document processor - handles PDFs properly"""
        
        # Determine file type
        file_extension = os.path.splitext(file_path)[1].lower()
        
        logger.info(f"Processing file: {file_path} (ext: {file_extension})")
        
        # FIXED: PDF files should be processed as documents, not images
        if file_extension == '.pdf':
            return await self._process_traditional_document(file_path, case_id)
        
        # Route to appropriate processor
        elif self._is_audio_file(file_extension):
            return await self.audio_processor.process_audio_file(file_path, case_id)
        
        elif self._is_video_file(file_extension):
            return await self.video_processor.process_video_file(file_path, case_id)
        
        elif self._is_image_file(file_extension):
            return await self.image_processor.process_image_file(file_path, case_id)
        
        else:
            # Process traditional documents
            return await self._process_traditional_document(file_path, case_id)
    
    def _is_audio_file(self, extension: str) -> bool:
        return extension in settings.supported_audio_formats
    
    def _is_video_file(self, extension: str) -> bool:
        return extension in settings.supported_video_formats
    
    def _is_image_file(self, extension: str) -> bool:
        # FIXED: Removed PDF from image formats
        image_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']
        return extension in image_formats
    
    async def _process_traditional_document(self, file_path: str, case_id: str) -> Dict[str, Any]:
        """Process traditional document formats with enhanced PDF support"""
        try:
            content = ""
            file_extension = os.path.splitext(file_path)[1].lower()
            
            logger.info(f"Processing {file_extension} document: {file_path}")
            
            if file_extension == '.pdf':
                content = await self._extract_pdf_text(file_path)
            elif file_extension == '.docx':
                content = await self._extract_docx_text(file_path)
            elif file_extension == '.txt':
                content = await self._extract_txt_text(file_path)
            else:
                # Try to read as text
                content = await self._extract_generic_text(file_path)
            
            # Extract entities and scan for privilege
            entities = await self._extract_document_entities(content)
            privilege_flags = await self._scan_document_privilege(content)
            
            # Generate summary
            summary = await self._generate_document_summary(content, entities, file_extension)
            
            logger.info(f"Successfully extracted {len(content)} characters from {file_path}")
            
            return {
                'id': f"doc_{case_id}_{os.path.basename(file_path).replace(' ', '_')}",
                'case_id': case_id,
                'file_path': file_path,
                'file_type': 'document',
                'content': content,
                'entities': entities,
                'privilege_flags': privilege_flags,
                'summary': summary,
                'metadata': {
                    'filename': os.path.basename(file_path),
                    'file_size': os.path.getsize(file_path),
                    'format': file_extension,
                    'content_length': len(content),
                    'word_count': len(content.split()) if content else 0
                },
                'processing_status': 'completed',
                'processed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                'id': f"doc_{case_id}_{os.path.basename(file_path).replace(' ', '_')}",
                'case_id': case_id,
                'file_path': file_path,
                'file_type': 'document',
                'content': f"Error processing document: {str(e)}",
                'error': str(e),
                'processing_status': 'failed',
                'processed_at': datetime.utcnow().isoformat()
            }
    
    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF using multiple methods"""
        content = ""
        
        # Method 1: Try PyPDF2 first
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                logger.info(f"PDF has {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            content += f"\n--- Page {page_num + 1} ---\n"
                            content += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                        
            if content.strip():
                logger.info(f"PyPDF2 extracted {len(content)} characters")
                return content.strip()
                
        except ImportError:
            logger.warning("PyPDF2 not available - install with: pip install PyPDF2")
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Method 2: Try pdfplumber if PyPDF2 fails
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                logger.info(f"PDF has {len(pdf.pages)} pages (pdfplumber)")
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            content += f"\n--- Page {page_num + 1} ---\n"
                            content += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1} with pdfplumber: {e}")
                        
            if content.strip():
                logger.info(f"pdfplumber extracted {len(content)} characters")
                return content.strip()
                
        except ImportError:
            logger.warning("pdfplumber not available - install with: pip install pdfplumber")
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # If all methods fail, return error message but still allow document to be indexed
        if not content.strip():
            logger.error(f"Failed to extract text from PDF: {file_path}")
            return f"PDF file: {os.path.basename(file_path)} - Text extraction failed. This may be a scanned PDF requiring OCR. File is available for search by filename."
        
        return content.strip()
    
    async def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            from docx import Document
            doc = Document(file_path)
            
            content = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content.append(paragraph.text)
            
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            content.append(cell.text)
            
            return "\n".join(content)
            
        except ImportError:
            logger.error("python-docx not available - install with: pip install python-docx")
            return f"Error: Cannot process DOCX file - python-docx not installed"
        except Exception as e:
            logger.error(f"Error processing DOCX: {e}")
            return f"Error processing DOCX file: {str(e)}"
    
    async def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                        logger.info(f"Successfully read TXT file with {encoding} encoding")
                        return content
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try binary mode
            with open(file_path, 'rb') as file:
                content = file.read()
                return content.decode('utf-8', errors='ignore')
                
        except Exception as e:
            logger.error(f"Error reading TXT file: {e}")
            return f"Error reading text file: {str(e)}"
    
    async def _extract_generic_text(self, file_path: str) -> str:
        """Try to extract text from unknown file types"""
        try:
            # Try reading as text with different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                        if content.strip():
                            return content
                except UnicodeDecodeError:
                    continue
            
            return f"Unable to extract text from {os.path.splitext(file_path)[1]} file"
            
        except Exception as e:
            logger.error(f"Error processing generic file: {e}")
            return f"Error processing file: {str(e)}"
    
    async def _extract_document_entities(self, content: str) -> Dict:
        """Extract entities from document content"""
        if not content:
            return {'legal_terms': [], 'case_numbers': [], 'dates': []}
        
        entities = {'legal_terms': [], 'case_numbers': [], 'dates': []}
        
        # Legal terms
        legal_terms = [
            'contract', 'agreement', 'lawsuit', 'plaintiff', 'defendant',
            'attorney', 'lawyer', 'counsel', 'court', 'judge', 'settlement',
            'damages', 'liability', 'negligence', 'breach', 'violation'
        ]
        
        content_lower = content.lower()
        for term in legal_terms:
            if term in content_lower:
                entities['legal_terms'].append(term)
        
        # Case numbers
        import re
        case_patterns = [
            r'\b\d{2,4}-\d{2,6}\b',
            r'\bCase No\.?\s*\d+\b',
            r'\bCivil No\.?\s*\d+\b'
        ]
        
        for pattern in case_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities['case_numbers'].extend(matches)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    async def _scan_document_privilege(self, content: str) -> List[Dict]:
        """Scan document for privilege indicators"""
        if not content:
            return []
        
        from app.utils.privilege_patterns import PrivilegeScanner
        scanner = PrivilegeScanner()
        return scanner.scan_text(content)
    
    async def _generate_document_summary(self, content: str, entities: Dict, file_format: str) -> str:
        """Generate document summary"""
        try:
            word_count = len(content.split()) if content else 0
            char_count = len(content) if content else 0
            
            summary_parts = [
                f"Document format: {file_format.upper()}",
                f"Content: {word_count} words, {char_count} characters"
            ]
            
            if entities.get('legal_terms'):
                summary_parts.append(f"Legal terms: {len(entities['legal_terms'])}")
            
            if entities.get('case_numbers'):
                summary_parts.append(f"Case numbers: {', '.join(entities['case_numbers'])}")
            
            return ". ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Document processed successfully"