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
        # Set tesseract path for Mac
        if os.path.exists(settings.tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path
        
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
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract image metadata: {str(e)}")
            return {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'format': os.path.splitext(file_path)[1].lower()
            }
    
    async def _perform_ocr(self, image_path: str) -> Dict:
        """Perform OCR using Tesseract"""
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Tesseract OCR
            try:
                text = pytesseract.image_to_string(
                    image,
                    config=f'--oem 3 --psm 6 -l {settings.ocr_language}'
                )
                
                # Get confidence data
                data = pytesseract.image_to_data(
                    image,
                    output_type=pytesseract.Output.DICT,
                    config=f'--oem 3 --psm 6 -l {settings.ocr_language}'
                )
                
                confidence = self._calculate_confidence(data)
                
            except Exception as e:
                logger.warning(f"Tesseract OCR failed: {str(e)}")
                text = ""
                confidence = 0.0
            
            return {
                'combined_text': text.strip(),
                'tesseract_text': text.strip(),
                'confidence': confidence,
                'total_confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error performing OCR: {str(e)}")
            return {
                'combined_text': '',
                'tesseract_text': '',
                'confidence': 0.0,
                'total_confidence': 0.0
            }
    
    def _calculate_confidence(self, data: Dict) -> float:
        """Calculate average confidence from Tesseract results"""
        if not data or 'conf' not in data:
            return 0.0
        
        confidences = [conf for conf in data['conf'] if conf > 0]
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    async def _extract_entities(self, text: str) -> Dict:
        """Extract entities from OCR text (simplified)"""
        if not text:
            return {'persons': [], 'organizations': [], 'legal_terms': []}
        
        # Simple keyword extraction
        legal_terms = []
        keywords = ['attorney', 'lawyer', 'court', 'judge', 'contract', 'agreement', 'settlement']
        
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                legal_terms.append(keyword)
        
        return {
            'persons': [],
            'organizations': [],
            'legal_terms': legal_terms
        }
    
    async def _generate_summary(self, metadata: Dict, ocr_results: Dict, entities: Dict) -> str:
        """Generate image summary"""
        try:
            summary_parts = [
                f"Image: {metadata.get('resolution', 'Unknown resolution')}",
                f"Format: {metadata.get('format', 'Unknown')}"
            ]
            
            if ocr_results.get('combined_text'):
                word_count = len(ocr_results['combined_text'].split())
                summary_parts.append(f"Text extracted: {word_count} words")
                summary_parts.append(f"OCR confidence: {ocr_results.get('confidence', 0):.1f}%")
            else:
                summary_parts.append("No text detected")
            
            if entities.get('legal_terms'):
                summary_parts.append(f"Legal terms: {', '.join(entities['legal_terms'])}")
            
            return ". ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Image processed successfully"

