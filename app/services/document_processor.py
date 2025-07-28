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
        """Enhanced document processor supporting multimedia - FIXED VERSION"""
    
    # Determine file type
        mime_type, _ = mimetypes.guess_type(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
    
        logger.info(f"Processing file: {file_path} (type: {mime_type}, ext: {file_extension})")
    
    # FIXED: Better file type detection
        if self._is_audio_file(file_extension, mime_type):
            return await self.audio_processor.process_audio_file(file_path, case_id)
    
        elif self._is_video_file(file_extension, mime_type):
            return await self.video_processor.process_video_file(file_path, case_id)
    
        elif self._is_image_file(file_extension, mime_type) or file_extension == '.pdf':
        # FIXED: Handle PDFs properly - they might be scanned documents
            if file_extension == '.pdf':
            # Try document processing first, then image processing if needed
                try:
                    result = await self._process_traditional_document(file_path, case_id)
                # If PDF has no extractable text, try OCR
                    if not result.get('content') or len(result.get('content', '').strip()) < 50:
                        logger.info(f"PDF {file_path} has little text, trying OCR...")
                        return await self.image_processor.process_image_file(file_path, case_id)
                    return result
                except Exception as e:
                    logger.warning(f"Document processing failed for PDF, trying OCR: {e}")
                    return await self.image_processor.process_image_file(file_path, case_id)
            else:
                return await self.image_processor.process_image_file(file_path, case_id)
    
        else:
        # Process traditional documents
            return await self._process_traditional_document(file_path, case_id) 
            await self._process_traditional_document(file_path, case_id)
    
    def _is_audio_file(self, extension: str, mime_type: str) -> bool:
        return (extension in settings.supported_audio_formats or 
                (mime_type and mime_type.startswith('audio/')))
    
    def _is_video_file(self, extension: str, mime_type: str) -> bool:
        return (extension in settings.supported_video_formats or 
                (mime_type and mime_type.startswith('video/')))
    
    def _is_image_file(self, extension: str, mime_type: str) -> bool:
        return (extension in settings.supported_image_formats or 
                (mime_type and mime_type.startswith('image/')))
    
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
                'id': f"doc_{case_id}_{os.path.basename(file_path)}",
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
                'id': f"doc_{case_id}_{os.path.basename(file_path)}",
                'case_id': case_id,
                'file_path': file_path,
                'file_type': 'document',
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
            logger.warning("PyPDF2 not available")
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
        
        # Method 3: Try using image OCR on PDF (for scanned PDFs)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            logger.info(f"PDF has {len(doc)} pages (PyMuPDF OCR fallback)")
            
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    
                    if text.strip():
                        content += f"\n--- Page {page_num + 1} ---\n"
                        content += text + "\n"
                    else:
                        # If no text, try OCR on the page image
                        logger.info(f"No text on page {page_num + 1}, trying OCR...")
                        pix = page.get_pixmap()
                        img_data = pix.tobytes("png")
                        
                        # Save temporary image and use OCR
                        temp_img_path = os.path.join(settings.temp_directory, f"temp_page_{page_num}.png")
                        with open(temp_img_path, "wb") as img_file:
                            img_file.write(img_data)
                        
                        # Use our image processor for OCR
                        ocr_result = await self.image_processor.process_image_file(temp_img_path, "temp")
                        if ocr_result.get('ocr_results', {}).get('combined_text'):
                            content += f"\n--- Page {page_num + 1} (OCR) ---\n"
                            content += ocr_result['ocr_results']['combined_text'] + "\n"
                        
                        # Clean up temp file
                        if os.path.exists(temp_img_path):
                            os.unlink(temp_img_path)
                            
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1} with PyMuPDF: {e}")
            
            doc.close()
            
            if content.strip():
                logger.info(f"PyMuPDF extracted {len(content)} characters")
                return content.strip()
                
        except ImportError:
            logger.warning("PyMuPDF not available - install with: pip install pymupdf")
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # If all methods fail
        if not content.strip():
            logger.error(f"Failed to extract text from PDF: {file_path}")
            return f"Error: Could not extract text from PDF file {os.path.basename(file_path)}. This may be a scanned PDF requiring OCR."
        
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