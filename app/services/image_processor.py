import pytesseract
from PIL import Image
import cv2
import numpy as np
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
from app.config import settings
from app.utils.privilege_patterns import PrivilegeScanner

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.privilege_scanner = PrivilegeScanner()
        self.ocr_available = self._setup_tesseract()
        
    def _setup_tesseract(self) -> bool:
        """Set up Tesseract OCR"""
        if not settings.tesseract_path:
            logger.warning("Tesseract not found. OCR functionality will be limited.")
            return False
        
        try:
            if settings.tesseract_path != "tesseract":  # Not just relying on PATH
                pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path
            
            # Test if Tesseract is working
            test_image = Image.new('RGB', (100, 30), color='white')
            pytesseract.image_to_string(test_image)
            logger.info(f"Tesseract OCR initialized at: {settings.tesseract_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Tesseract setup failed: {str(e)}. OCR will be limited.")
            return False
        
    async def process_image_file(self, file_path: str, case_id: str) -> Dict:
        """Process image file for legal discovery"""
        logger.info(f"Processing image file: {file_path}")
        
        try:
            # Extract metadata
            metadata = await self._extract_image_metadata(file_path)
            
            # Perform OCR
            ocr_results = await self._perform_ocr(file_path)
            
            # Scan for privilege
            privilege_flags = []
            if ocr_results.get('combined_text'):
                privilege_flags = self.privilege_scanner.scan_text(ocr_results['combined_text'])
            
            # Extract entities
            entities = await self._extract_entities(ocr_results.get('combined_text', ''))
            
            # Generate summary
            summary = await self._generate_summary(metadata, ocr_results, entities)
            
            return {
                'id': f"image_{case_id}_{os.path.basename(file_path)}",
                'case_id': case_id,
                'file_path': file_path,
                'file_type': 'image',
                'metadata': metadata,
                'ocr_results': ocr_results,
                'privilege_flags': privilege_flags,
                'entities': entities,
                'summary': summary,
                'processing_status': 'completed',
                'processed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {str(e)}")
            return {
                'id': f"image_{case_id}_{os.path.basename(file_path)}",
                'case_id': case_id,
                'file_path': file_path,
                'file_type': 'image',
                'error': str(e),
                'processing_status': 'failed',
                'processed_at': datetime.utcnow().isoformat()
            }
    
    async def _extract_image_metadata(self, file_path: str) -> Dict:
        """Extract image metadata"""
        try:
            image = Image.open(file_path)
            
            metadata = {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'format': image.format,
                'mode': image.mode,
                'width': image.width,
                'height': image.height,
                'resolution': f"{image.width}x{image.height}"
            }
            
            # Try to get EXIF data
            try:
                exif = image._getexif()
                if exif:
                    metadata['exif_data'] = {}
                    for tag_id, value in exif.items():
                        if isinstance(value, (str, int, float)):
                            metadata['exif_data'][str(tag_id)] = value
            except:
                pass
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract image metadata: {str(e)}")
            return {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'format': os.path.splitext(file_path)[1].lower(),
                'error': str(e)
            }
    
    async def _perform_ocr(self, image_path: str) -> Dict:
        """Perform OCR using multiple methods"""
        result = {
            'combined_text': '',
            'tesseract_text': '',
            'confidence': 0.0,
            'total_confidence': 0.0,
            'method_used': 'none'
        }
        
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Try Tesseract first
            if self.ocr_available:
                try:
                    text = pytesseract.image_to_string(
                        image,
                        config=f'--oem 3 --psm 6 -l {settings.ocr_language}'
                    )
                    
                    if text.strip():
                        # Get confidence data
                        data = pytesseract.image_to_data(
                            image,
                            output_type=pytesseract.Output.DICT,
                            config=f'--oem 3 --psm 6 -l {settings.ocr_language}'
                        )
                        
                        confidence = self._calculate_confidence(data)
                        
                        result.update({
                            'combined_text': text.strip(),
                            'tesseract_text': text.strip(),
                            'confidence': confidence,
                            'total_confidence': confidence,
                            'method_used': 'tesseract'
                        })
                        
                        logger.info(f"Tesseract OCR successful with {confidence:.1f}% confidence")
                        return result
                        
                except Exception as e:
                    logger.warning(f"Tesseract OCR failed: {str(e)}")
            
            # Try EasyOCR as fallback
            try:
                import easyocr
                reader = easyocr.Reader(['en'])
                ocr_result = reader.readtext(image_path)
                
                if ocr_result:
                    texts = []
                    confidences = []
                    
                    for detection in ocr_result:
                        text = detection[1]
                        confidence = detection[2]
                        
                        if confidence > 0.5:  # Only include high-confidence text
                            texts.append(text)
                            confidences.append(confidence)
                    
                    if texts:
                        combined_text = ' '.join(texts)
                        avg_confidence = sum(confidences) / len(confidences) * 100
                        
                        result.update({
                            'combined_text': combined_text,
                            'easyocr_text': combined_text,
                            'confidence': avg_confidence,
                            'total_confidence': avg_confidence,
                            'method_used': 'easyocr'
                        })
                        
                        logger.info(f"EasyOCR successful with {avg_confidence:.1f}% confidence")
                        return result
                        
            except ImportError:
                logger.info("EasyOCR not available")
            except Exception as e:
                logger.warning(f"EasyOCR failed: {str(e)}")
            
            # If both methods fail, try basic OpenCV preprocessing + Tesseract
            if self.ocr_available:
                try:
                    processed_image = self._preprocess_image_for_ocr(image_path)
                    
                    text = pytesseract.image_to_string(
                        processed_image,
                        config='--oem 3 --psm 6'
                    )
                    
                    if text.strip():
                        result.update({
                            'combined_text': text.strip(),
                            'tesseract_text': text.strip(),
                            'confidence': 50.0,  # Assume lower confidence for preprocessed
                            'total_confidence': 50.0,
                            'method_used': 'tesseract_preprocessed'
                        })
                        logger.info("OCR successful with image preprocessing")
                        return result
                        
                except Exception as e:
                    logger.warning(f"Preprocessed OCR failed: {str(e)}")
            
            logger.warning(f"All OCR methods failed for {image_path}")
            result['method_used'] = 'failed'
            return result
            
        except Exception as e:
            logger.error(f"Error performing OCR: {str(e)}")
            result['error'] = str(e)
            return result
    
    def _preprocess_image_for_ocr(self, image_path: str):
        """Preprocess image to improve OCR accuracy"""
        try:
            # Read image with OpenCV
            img = cv2.imread(image_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply thresholding to get black text on white background
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL Image
            return Image.fromarray(thresh)
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}")
            return Image.open(image_path)
    
    def _calculate_confidence(self, data: Dict) -> float:
        """Calculate average confidence from Tesseract results"""
        if not data or 'conf' not in data:
            return 0.0
        
        confidences = [conf for conf in data['conf'] if conf > 0]
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    async def _extract_entities(self, text: str) -> Dict:
        """Extract entities from OCR text"""
        if not text:
            return {
                'persons': [], 'organizations': [], 'legal_terms': [], 
                'dates': [], 'phone_numbers': [], 'email_addresses': []
            }
        
        entities = {
            'persons': [],
            'organizations': [],
            'legal_terms': [],
            'dates': [],
            'phone_numbers': [],
            'email_addresses': []
        }
        
        # Extract legal terms
        legal_terms = [
            'attorney', 'lawyer', 'court', 'judge', 'contract', 'agreement', 
            'settlement', 'plaintiff', 'defendant', 'evidence', 'testimony',
            'deposition', 'subpoena', 'hearing', 'trial', 'verdict', 'appeal'
        ]
        
        text_lower = text.lower()
        for term in legal_terms:
            if term in text_lower:
                entities['legal_terms'].append(term)
        
        # Extract dates using regex
        import re
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',          # MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',          # MM-DD-YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',          # YYYY-MM-DD
            r'[A-Za-z]+ \d{1,2}, \d{4}',       # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            entities['dates'].extend(matches)
        
        # Extract phone numbers
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
        phone_matches = re.findall(phone_pattern, text)
        entities['phone_numbers'] = ['-'.join(match) for match in phone_matches]
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['email_addresses'] = re.findall(email_pattern, text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    async def _generate_summary(self, metadata: Dict, ocr_results: Dict, entities: Dict) -> str:
        """Generate image summary"""
        try:
            summary_parts = []
            
            # Basic image info
            resolution = metadata.get('resolution', 'Unknown resolution')
            format_type = metadata.get('format', 'Unknown')
            summary_parts.append(f"Image: {resolution}, {format_type} format")
            
            # OCR results
            if ocr_results.get('combined_text'):
                word_count = len(ocr_results['combined_text'].split())
                confidence = ocr_results.get('confidence', 0)
                method = ocr_results.get('method_used', 'unknown')
                summary_parts.append(f"Text extracted: {word_count} words ({confidence:.1f}% confidence, {method})")
            else:
                summary_parts.append("No text detected in image")
            
            # Entities
            if entities.get('legal_terms'):
                summary_parts.append(f"Legal terms: {', '.join(entities['legal_terms'][:3])}")
            
            if entities.get('dates'):
                summary_parts.append(f"Dates found: {len(entities['dates'])}")
            
            if entities.get('phone_numbers'):
                summary_parts.append(f"Phone numbers: {len(entities['phone_numbers'])}")
            
            if entities.get('email_addresses'):
                summary_parts.append(f"Email addresses: {len(entities['email_addresses'])}")
            
            return ". ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Image processed successfully"