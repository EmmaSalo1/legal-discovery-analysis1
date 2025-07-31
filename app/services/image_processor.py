import os
import logging
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
import json

# OCR engines
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

from app.config import settings
from app.utils.privilege_patterns import PrivilegeScanner

logger = logging.getLogger(__name__)

class EnhancedImageProcessor:
    def __init__(self):
        self.privilege_scanner = PrivilegeScanner()
        
        # Initialize OCR engines
        self.ocr_engines = self._initialize_ocr_engines()
        logger.info(f"Available OCR engines: {list(self.ocr_engines.keys())}")
        
        # Initialize EasyOCR reader if available
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
                logger.info("EasyOCR reader initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
    
    def _initialize_ocr_engines(self) -> Dict[str, bool]:
        """Initialize and check available OCR engines"""
        engines = {}
        
        # Check Tesseract
        if TESSERACT_AVAILABLE:
            try:
                if os.path.exists(settings.tesseract_path):
                    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path
                
                # Test Tesseract
                test_result = pytesseract.image_to_string(
                    Image.new('RGB', (100, 50), color='white')
                )
                engines['tesseract'] = True
                logger.info("Tesseract OCR available")
            except Exception as e:
                engines['tesseract'] = False
                logger.warning(f"Tesseract not available: {e}")
        else:
            engines['tesseract'] = False
        
        # Check EasyOCR
        engines['easyocr'] = EASYOCR_AVAILABLE
        
        return engines
    
    async def process_image_file(self, file_path: str, case_id: str) -> Dict:
        """Enhanced image processing with multiple OCR engines and visual analysis"""
        logger.info(f"Processing image file: {file_path}")
        
        try:
            # Load and analyze image
            image = Image.open(file_path)
            
            # Extract comprehensive metadata
            metadata = await self._extract_enhanced_metadata(file_path, image)
            
            # Preprocess image for better OCR
            preprocessed_images = await self._preprocess_image_for_ocr(image)
            
            # Perform OCR with multiple engines
            ocr_results = await self._perform_comprehensive_ocr(preprocessed_images)
            
            # Visual content analysis
            visual_analysis = await self._analyze_visual_content(image, file_path)
            
            # Document structure analysis
            document_analysis = await self._analyze_document_structure(image, ocr_results)
            
            # Extract structured information
            structured_data = await self._extract_structured_information(ocr_results['combined_text'])
            
            # Scan for privilege
            privilege_flags = []
            if ocr_results.get('combined_text'):
                privilege_flags = self.privilege_scanner.scan_text(ocr_results['combined_text'])
            
            # Generate comprehensive summary
            summary = await self._generate_enhanced_summary(
                metadata, ocr_results, visual_analysis, document_analysis, structured_data
            )
            
            result = {
                'id': f"image_{case_id}_{os.path.basename(file_path)}",
                'case_id': case_id,
                'file_path': file_path,
                'file_type': 'image',
                'metadata': metadata,
                'ocr_results': ocr_results,
                'visual_analysis': visual_analysis,
                'document_analysis': document_analysis,
                'structured_data': structured_data,
                'privilege_flags': privilege_flags,
                'summary': summary,
                'processing_status': 'completed',
                'processed_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Successfully processed image: {file_path}")
            return result
            
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
    
    async def _extract_enhanced_metadata(self, file_path: str, image: Image.Image) -> Dict:
        """Extract comprehensive image metadata"""
        try:
            metadata = {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'format': image.format,
                'mode': image.mode,
                'width': image.width,
                'height': image.height,
                'resolution': f"{image.width}x{image.height}",
                'aspect_ratio': round(image.width / image.height, 2) if image.height > 0 else 0,
                'color_depth': len(image.getbands()) if hasattr(image, 'getbands') else 1
            }
            
            # Extract EXIF data if available
            exif_data = {}
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif:
                    # Common EXIF tags
                    exif_tags = {
                        'DateTime': 306,
                        'Software': 305,
                        'Make': 271,
                        'Model': 272,
                        'XResolution': 282,
                        'YResolution': 283
                    }
                    
                    for tag_name, tag_id in exif_tags.items():
                        if tag_id in exif:
                            exif_data[tag_name] = str(exif[tag_id])
            
            if exif_data:
                metadata['exif'] = exif_data
            
            # Analyze image characteristics
            metadata['estimated_dpi'] = self._estimate_dpi(image)
            metadata['is_scanned_document'] = self._is_likely_scanned_document(image)
            metadata['image_type'] = self._classify_image_type(image)
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract enhanced metadata: {str(e)}")
            return {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'format': os.path.splitext(file_path)[1].lower()
            }
    
    def _estimate_dpi(self, image: Image.Image) -> int:
        """Estimate DPI of the image"""
        try:
            if hasattr(image, 'info') and 'dpi' in image.info:
                return image.info['dpi'][0]
            
            # Estimate based on image size (rough heuristic)
            if image.width > 2000 and image.height > 2000:
                return 300  # High resolution
            elif image.width > 1000 and image.height > 1000:
                return 150  # Medium resolution
            else:
                return 72   # Low resolution/screen
                
        except:
            return 150  # Default estimate
    
    def _is_likely_scanned_document(self, image: Image.Image) -> bool:
        """Determine if image is likely a scanned document"""
        try:
            # Convert to grayscale for analysis
            gray = image.convert('L')
            
            # Check aspect ratio (documents are often rectangular)
            aspect_ratio = image.width / image.height
            is_document_ratio = 0.5 < aspect_ratio < 2.0
            
            # Check if image is mostly white/light (typical for documents)
            histogram = gray.histogram()
            light_pixels = sum(histogram[200:])  # Pixels with brightness > 200
            total_pixels = image.width * image.height
            is_mostly_light = (light_pixels / total_pixels) > 0.6
            
            # Check for high contrast (text on background)
            extremes = histogram[:50] + histogram[205:]  # Very dark + very light
            has_high_contrast = (sum(extremes) / total_pixels) > 0.7
            
            return is_document_ratio and is_mostly_light and has_high_contrast
            
        except:
            return False
    
    def _classify_image_type(self, image: Image.Image) -> str:
        """Classify the type of image"""
        try:
            if self._is_likely_scanned_document(image):
                return 'scanned_document'
            
            # Check for screenshots (typically have specific aspect ratios)
            aspect_ratio = image.width / image.height
            if abs(aspect_ratio - 16/9) < 0.1 or abs(aspect_ratio - 4/3) < 0.1:
                return 'screenshot'
            
            # Check for photos (usually have more color variety)
            if image.mode in ['RGB', 'RGBA']:
                colors = image.getcolors(maxcolors=256*256*256)
                if colors and len(colors) > 1000:
                    return 'photograph'
            
            return 'general_image'
            
        except:
            return 'unknown'
    
    async def _preprocess_image_for_ocr(self, image: Image.Image) -> Dict[str, Image.Image]:
        """Create multiple preprocessed versions for better OCR"""
        try:
            preprocessed = {'original': image}
            
            # Convert to grayscale
            gray = image.convert('L')
            preprocessed['grayscale'] = gray
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(gray)
            high_contrast = enhancer.enhance(2.0)
            preprocessed['high_contrast'] = high_contrast
            
            # Sharpen
            sharpened = gray.filter(ImageFilter.SHARPEN)
            preprocessed['sharpened'] = sharpened
            
            # Threshold (black and white)
            threshold = gray.point(lambda x: 0 if x < 128 else 255)
            preprocessed['threshold'] = threshold
            
            # Noise reduction
            denoised = gray.filter(ImageFilter.MedianFilter(3))
            preprocessed['denoised'] = denoised
            
            # Scale up if image is small (improves OCR accuracy)
            if image.width < 1000 or image.height < 1000:
                scale_factor = 2
                scaled = gray.resize(
                    (image.width * scale_factor, image.height * scale_factor),
                    Image.LANCZOS
                )
                preprocessed['scaled'] = scaled
            
            return preprocessed
            
        except Exception as e:
            logger.warning(f"Preprocessing error: {e}")
            return {'original': image}
    
    async def _perform_comprehensive_ocr(self, preprocessed_images: Dict[str, Image.Image]) -> Dict:
        """Perform OCR using multiple engines and preprocessing variants"""
        ocr_results = {
            'combined_text': '',
            'engine_results': {},
            'confidence_scores': {},
            'best_result': '',
            'total_confidence': 0.0
        }
        
        try:
            all_texts = []
            all_confidences = []
            
            # Try each preprocessing variant with each available OCR engine
            for prep_name, image in preprocessed_images.items():
                
                # Tesseract OCR
                if self.ocr_engines.get('tesseract'):
                    try:
                        result = await self._tesseract_ocr(image, prep_name)
                        if result['text'].strip():
                            all_texts.append(result['text'])
                            all_confidences.append(result['confidence'])
                            ocr_results['engine_results'][f'tesseract_{prep_name}'] = result
                    except Exception as e:
                        logger.warning(f"Tesseract OCR failed for {prep_name}: {e}")
                
                # EasyOCR
                if self.ocr_engines.get('easyocr') and self.easyocr_reader:
                    try:
                        result = await self._easyocr_ocr(image, prep_name)
                        if result['text'].strip():
                            all_texts.append(result['text'])
                            all_confidences.append(result['confidence'])
                            ocr_results['engine_results'][f'easyocr_{prep_name}'] = result
                    except Exception as e:
                        logger.warning(f"EasyOCR failed for {prep_name}: {e}")
            
            # Combine results
            if all_texts:
                # Find the best result (highest confidence)
                best_idx = all_confidences.index(max(all_confidences))
                ocr_results['best_result'] = all_texts[best_idx]
                ocr_results['total_confidence'] = max(all_confidences)
                
                # Combine all unique text
                unique_lines = set()
                for text in all_texts:
                    for line in text.split('\n'):
                        line = line.strip()
                        if line:
                            unique_lines.add(line)
                
                ocr_results['combined_text'] = '\n'.join(sorted(unique_lines))
                
                # Calculate average confidence
                if all_confidences:
                    ocr_results['average_confidence'] = sum(all_confidences) / len(all_confidences)
            
            logger.info(f"OCR completed with {len(all_texts)} results, best confidence: {ocr_results['total_confidence']:.1f}")
            return ocr_results
            
        except Exception as e:
            logger.error(f"Comprehensive OCR error: {e}")
            return ocr_results
    
    async def _tesseract_ocr(self, image: Image.Image, prep_name: str) -> Dict:
        """Perform OCR using Tesseract"""
        try:
            # Get text with word-level confidence
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config=f'--oem 3 --psm 6 -l {settings.ocr_language}'
            )
            
            # Extract text
            text = pytesseract.image_to_string(
                image,
                config=f'--oem 3 --psm 6 -l {settings.ocr_language}'
            )
            
            # Calculate confidence
            confidences = [conf for conf in data['conf'] if conf > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': text.strip(),
                'confidence': avg_confidence,
                'engine': 'tesseract',
                'preprocessing': prep_name,
                'word_count': len(text.split()),
                'char_count': len(text)
            }
            
        except Exception as e:
            logger.warning(f"Tesseract OCR error: {e}")
            return {'text': '', 'confidence': 0, 'engine': 'tesseract', 'error': str(e)}
    
    async def _easyocr_ocr(self, image: Image.Image, prep_name: str) -> Dict:
        """Perform OCR using EasyOCR"""
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Perform OCR
            results = self.easyocr_reader.readtext(img_array)
            
            # Extract text and confidence
            text_parts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low-confidence results
                    text_parts.append(text)
                    confidences.append(confidence)
            
            combined_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': combined_text.strip(),
                'confidence': avg_confidence * 100,  # Convert to percentage
                'engine': 'easyocr',
                'preprocessing': prep_name,
                'word_count': len(combined_text.split()),
                'char_count': len(combined_text),
                'detection_count': len(results)
            }
            
        except Exception as e:
            logger.warning(f"EasyOCR error: {e}")
            return {'text': '', 'confidence': 0, 'engine': 'easyocr', 'error': str(e)}
    
    async def _analyze_visual_content(self, image: Image.Image, file_path: str) -> Dict:
        """Analyze visual content of the image"""
        try:
            analysis = {
                'layout_type': 'unknown',
                'text_regions': [],
                'visual_elements': {},
                'color_analysis': {},
                'quality_assessment': {}
            }
            
            # Convert to OpenCV format for analysis
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect text regions
            text_regions = await self._detect_text_regions(gray)
            analysis['text_regions'] = text_regions
            
            # Analyze layout
            analysis['layout_type'] = await self._analyze_layout(gray, text_regions)
            
            # Detect visual elements
            analysis['visual_elements'] = await self._detect_visual_elements(cv_image, gray)
            
            # Color analysis
            analysis['color_analysis'] = await self._analyze_colors(image)
            
            # Quality assessment
            analysis['quality_assessment'] = await self._assess_image_quality(gray)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Visual content analysis error: {e}")
            return {'error': str(e)}
    
    async def _detect_text_regions(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect text regions in the image"""
        try:
            text_regions = []
            
            # Use MSER (Maximally Stable Extremal Regions) to detect text
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray_image)
            
            # Filter and analyze regions
            for region in regions:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                
                # Filter based on size and aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                if (10 < w < gray_image.shape[1] * 0.8 and 
                    5 < h < gray_image.shape[0] * 0.3 and
                    0.1 < aspect_ratio < 20 and
                    area > 50):
                    
                    text_regions.append({
                        'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h),
                        'area': int(area),
                        'aspect_ratio': round(aspect_ratio, 2)
                    })
            
            # Sort by position (top to bottom, left to right)
            text_regions.sort(key=lambda r: (r['y'], r['x']))
            
            return text_regions[:20]  # Limit to top 20 regions
            
        except Exception as e:
            logger.warning(f"Text region detection error: {e}")
            return []
    
    async def _analyze_layout(self, gray_image: np.ndarray, text_regions: List[Dict]) -> str:
        """Analyze document layout type"""
        try:
            if not text_regions:
                return 'no_text'
            
            # Calculate layout characteristics
            height, width = gray_image.shape
            
            # Group regions by approximate rows
            rows = {}
            for region in text_regions:
                row_key = region['y'] // 20  # Group by 20-pixel rows
                if row_key not in rows:
                    rows[row_key] = []
                rows[row_key].append(region)
            
            # Analyze patterns
            avg_regions_per_row = len(text_regions) / len(rows) if rows else 0
            
            # Check for column layout
            left_regions = sum(1 for r in text_regions if r['x'] < width * 0.3)
            right_regions = sum(1 for r in text_regions if r['x'] > width * 0.7)
            center_regions = len(text_regions) - left_regions - right_regions
            
            # Classify layout
            if avg_regions_per_row > 3:
                return 'table_or_form'
            elif left_regions > 0 and right_regions > 0 and center_regions < len(text_regions) * 0.3:
                return 'two_column'
            elif len(rows) > 10 and avg_regions_per_row < 2:
                return 'single_column_document'
            elif len(text_regions) < 5:
                return 'sparse_text'
            else:
                return 'mixed_layout'
                
        except Exception as e:
            logger.warning(f"Layout analysis error: {e}")
            return 'unknown'
    
    async def _detect_visual_elements(self, cv_image: np.ndarray, gray_image: np.ndarray) -> Dict:
        """Detect various visual elements"""
        try:
            elements = {
                'lines': 0,
                'rectangles': 0,
                'circles': 0,
                'logos_or_stamps': 0,
                'signatures': 0,
                'charts_or_graphs': 0
            }
            
            # Detect lines using HoughLines
            edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            elements['lines'] = len(lines) if lines is not None else 0
            
            # Detect rectangles and circles using contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter small contours
                    # Approximate contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) == 4:
                        elements['rectangles'] += 1
                    elif len(approx) > 8:
                        # Could be a circle or complex shape
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        if abs(area - np.pi * radius * radius) / area < 0.2:
                            elements['circles'] += 1
            
            # Simple signature detection (look for curved, isolated regions)
            # This is a basic heuristic
            isolated_regions = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 < area < 2000:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0 and area / hull_area < 0.7:  # Concave shape
                        isolated_regions += 1
            
            elements['signatures'] = min(isolated_regions, 3)  # Cap at 3
            
            return elements
            
        except Exception as e:
            logger.warning(f"Visual element detection error: {e}")
            return {}
    
    async def _analyze_colors(self, image: Image.Image) -> Dict:
        """Analyze color characteristics of the image"""
        try:
            analysis = {
                'is_grayscale': False,
                'dominant_colors': [],
                'color_distribution': {},
                'background_color': None
            }
            
            # Check if effectively grayscale
            if image.mode in ['L', '1']:
                analysis['is_grayscale'] = True
            elif image.mode in ['RGB', 'RGBA']:
                # Sample pixels to check color variance
                pixels = np.array(image)
                if len(pixels.shape) == 3:
                    r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
                    color_variance = np.var([np.var(r), np.var(g), np.var(b)])
                    analysis['is_grayscale'] = color_variance < 100
            
            # Get dominant colors using quantization
            if not analysis['is_grayscale'] and image.mode in ['RGB', 'RGBA']:
                # Convert to RGB if needed
                rgb_image = image.convert('RGB')
                
                # Quantize colors
                quantized = rgb_image.quantize(colors=8, method=Image.MEDIANCUT)
                palette = quantized.getpalette()
                
                # Extract dominant colors
                dominant_colors = []
                for i in range(0, min(24, len(palette)), 3):
                    color = [palette[i], palette[i+1], palette[i+2]]
                    dominant_colors.append(color)
                
                analysis['dominant_colors'] = dominant_colors[:5]
                
                # Estimate background color (most common color at edges)
                edge_pixels = []
                width, height = rgb_image.size
                
                # Sample edge pixels
                for x in [0, width-1]:
                    for y in range(0, height, max(1, height//20)):
                        try:
                            edge_pixels.append(rgb_image.getpixel((x, y)))
                        except:
                            pass
                
                for y in [0, height-1]:
                    for x in range(0, width, max(1, width//20)):
                        try:
                            edge_pixels.append(rgb_image.getpixel((x, y)))
                        except:
                            pass
                
                if edge_pixels:
                    # Find most common edge color
                    from collections import Counter
                    color_counts = Counter(edge_pixels)
                    analysis['background_color'] = list(color_counts.most_common(1)[0][0])
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Color analysis error: {e}")
            return {}
    
    async def _assess_image_quality(self, gray_image: np.ndarray) -> Dict:
        """Assess image quality for OCR purposes"""
        try:
            assessment = {
                'overall_quality': 'medium',
                'sharpness_score': 0.0,
                'contrast_score': 0.0,
                'noise_level': 0.0,
                'resolution_adequacy': 'medium'
            }
            
            # Sharpness assessment using Laplacian variance
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            sharpness = laplacian.var()
            assessment['sharpness_score'] = min(1.0, sharpness / 1000)  # Normalize
            
            # Contrast assessment
            contrast = gray_image.std()
            assessment['contrast_score'] = min(1.0, contrast / 64)  # Normalize
            
            # Noise level estimation
            noise = cv2.fastNlMeansDenoising(gray_image)
            noise_level = np.mean(np.abs(gray_image.astype(float) - noise.astype(float)))
            assessment['noise_level'] = min(1.0, noise_level / 10)  # Normalize
            
            # Resolution adequacy
            height, width = gray_image.shape
            if width >= 1500 and height >= 1500:
                assessment['resolution_adequacy'] = 'high'
            elif width >= 800 and height >= 800:
                assessment['resolution_adequacy'] = 'medium'
            else:
                assessment['resolution_adequacy'] = 'low'
            
            # Overall quality score
            quality_score = (
                assessment['sharpness_score'] * 0.4 +
                assessment['contrast_score'] * 0.3 +
                (1 - assessment['noise_level']) * 0.3
            )
            
            if quality_score > 0.7:
                assessment['overall_quality'] = 'high'
            elif quality_score > 0.4:
                assessment['overall_quality'] = 'medium'
            else:
                assessment['overall_quality'] = 'low'
            
            return assessment
            
        except Exception as e:
            logger.warning(f"Quality assessment error: {e}")
            return {'overall_quality': 'unknown'}
    
    async def _analyze_document_structure(self, image: Image.Image, ocr_results: Dict) -> Dict:
        """Analyze document structure and type"""
        try:
            structure = {
                'document_type': 'unknown',
                'has_header': False,
                'has_footer': False,
                'has_letterhead': False,
                'has_signature_area': False,
                'table_detected': False,
                'form_fields': [],
                'confidence': 0.5
            }
            
            text = ocr_results.get('combined_text', '')
            if not text:
                return structure
            
            text_lower = text.lower()
            
            # Document type detection
            if any(word in text_lower for word in ['contract', 'agreement', 'terms', 'party']):
                structure['document_type'] = 'contract'
            elif any(word in text_lower for word in ['invoice', 'bill', 'payment', 'due']):
                structure['document_type'] = 'invoice'
            elif any(word in text_lower for word in ['memo', 'memorandum', 'subject:', 'from:']):
                structure['document_type'] = 'memo'
            elif any(word in text_lower for word in ['dear', 'sincerely', 'yours truly']):
                structure['document_type'] = 'letter'
            elif any(word in text_lower for word in ['deposition', 'testimony', 'sworn']):
                structure['document_type'] = 'legal_document'
            elif any(word in text_lower for word in ['email', '@', 'subject:', 'sent:']):
                structure['document_type'] = 'email'
            
            # Structure elements detection
            lines = text.split('\n')
            
            # Header detection (look for titles, letterheads in first few lines)
            if len(lines) > 0:
                first_lines = ' '.join(lines[:3]).lower()
                if any(word in first_lines for word in ['company', 'corporation', 'llc', 'inc']):
                    structure['has_letterhead'] = True
                    structure['has_header'] = True
            
            # Footer detection (look for page numbers, contact info in last few lines)
            if len(lines) > 5:
                last_lines = ' '.join(lines[-3:]).lower()
                if any(word in last_lines for word in ['page', 'phone', 'email', 'address']):
                    structure['has_footer'] = True
            
            # Signature area detection
            if any(word in text_lower for word in ['signature', 'signed', 'date:', 'name:']):
                structure['has_signature_area'] = True
            
            # Table detection (look for structured data patterns)
            table_indicators = text_lower.count('|') + text_lower.count('\t')
            repeated_patterns = len(re.findall(r'(\b\w+\b).*?\1', text_lower))
            
            if table_indicators > 10 or repeated_patterns > 5:
                structure['table_detected'] = True
            
            # Form fields detection
            form_patterns = [
                r'name[:\s]*_+',
                r'date[:\s]*_+',
                r'signature[:\s]*_+',
                r'\[\s*\]',  # Checkboxes
                r'_+\s*(?:name|date|signature)',
            ]
            
            for pattern in form_patterns:
                matches = re.findall(pattern, text_lower)
                structure['form_fields'].extend(matches)
            
            # Calculate confidence based on detected features
            feature_count = sum([
                structure['has_header'],
                structure['has_footer'],
                structure['has_letterhead'],
                structure['has_signature_area'],
                structure['table_detected'],
                len(structure['form_fields']) > 0,
                structure['document_type'] != 'unknown'
            ])
            
            structure['confidence'] = min(1.0, feature_count / 5.0)
            
            return structure
            
        except Exception as e:
            logger.warning(f"Document structure analysis error: {e}")
            return {'document_type': 'unknown', 'confidence': 0.0}
    
    async def _extract_structured_information(self, text: str) -> Dict:
        """Extract structured information from text"""
        try:
            structured = {
                'dates': [],
                'names': [],
                'addresses': [],
                'phone_numbers': [],
                'email_addresses': [],
                'monetary_amounts': [],
                'case_numbers': [],
                'legal_terms': []
            }
            
            if not text:
                return structured
            
            # Extract dates
            date_patterns = [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b'
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                structured['dates'].extend(matches)
            
            # Extract phone numbers
            phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
            phone_matches = re.findall(phone_pattern, text)
            structured['phone_numbers'] = ['-'.join(match) for match in phone_matches]
            
            # Extract email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            structured['email_addresses'] = re.findall(email_pattern, text)
            
            # Extract monetary amounts
            money_patterns = [
                r'\$[\d,]+\.?\d*',
                r'\b\d+\.\d{2}\s*dollars?\b',
                r'\b\d+\s*USD\b'
            ]
            
            for pattern in money_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                structured['monetary_amounts'].extend(matches)
            
            # Extract case numbers
            case_patterns = [
                r'\b\d{2,4}-\d{2,6}\b',
                r'\bCase No\.?\s*\d+\b',
                r'\bCivil No\.?\s*\d+\b',
                r'\bDocket No\.?\s*\d+\b'
            ]
            
            for pattern in case_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                structured['case_numbers'].extend(matches)
            
            # Extract names (simple heuristic - capitalized words)
            name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
            potential_names = re.findall(name_pattern, text)
            
            # Filter out common words that aren't names
            common_words = {
                'United States', 'New York', 'Los Angeles', 'Court Order',
                'Case Number', 'Legal Notice', 'Page Number'
            }
            
            structured['names'] = [name for name in potential_names if name not in common_words][:10]
            
            # Extract legal terms
            legal_terms = [
                'plaintiff', 'defendant', 'attorney', 'counsel', 'court',
                'judge', 'settlement', 'damages', 'liability', 'negligence',
                'breach', 'contract', 'agreement', 'deposition', 'testimony',
                'evidence', 'witness', 'objection', 'sustained', 'overruled'
            ]
            
            text_lower = text.lower()
            found_terms = [term for term in legal_terms if term in text_lower]
            structured['legal_terms'] = list(set(found_terms))
            
            # Remove duplicates and empty entries
            for key in structured:
                if isinstance(structured[key], list):
                    structured[key] = list(set(filter(None, structured[key])))
            
            return structured
            
        except Exception as e:
            logger.warning(f"Structured information extraction error: {e}")
            return {}
    
    async def _generate_enhanced_summary(self, metadata: Dict, ocr_results: Dict, 
                                       visual_analysis: Dict, document_analysis: Dict, 
                                       structured_data: Dict) -> str:
        """Generate comprehensive image summary"""
        try:
            summary_parts = []
            
            # Basic image info
            resolution = metadata.get('resolution', 'Unknown')
            file_format = metadata.get('format', 'Unknown')
            summary_parts.append(f"Image: {resolution}, {file_format} format")
            
            # Image type and quality
            image_type = metadata.get('image_type', 'unknown')
            if image_type != 'unknown':
                summary_parts.append(f"Type: {image_type.replace('_', ' ')}")
            
            quality = visual_analysis.get('quality_assessment', {}).get('overall_quality', 'unknown')
            if quality != 'unknown':
                summary_parts.append(f"Image quality: {quality}")
            
            # OCR results
            combined_text = ocr_results.get('combined_text', '')
            if combined_text:
                word_count = len(combined_text.split())
                confidence = ocr_results.get('total_confidence', 0)
                summary_parts.append(f"Text extracted: {word_count} words ({confidence:.1f}% confidence)")
                
                # Document type
                doc_type = document_analysis.get('document_type', 'unknown')
                if doc_type != 'unknown':
                    summary_parts.append(f"Document type: {doc_type}")
            else:
                summary_parts.append("No text detected")
            
            # Structured data highlights
            if structured_data:
                highlights = []
                if structured_data.get('dates'):
                    highlights.append(f"{len(structured_data['dates'])} dates")
                if structured_data.get('monetary_amounts'):
                    highlights.append(f"{len(structured_data['monetary_amounts'])} monetary amounts")
                if structured_data.get('legal_terms'):
                    highlights.append(f"{len(structured_data['legal_terms'])} legal terms")
                
                if highlights:
                    summary_parts.append(f"Found: {', '.join(highlights)}")
            
            # Visual elements
            visual_elements = visual_analysis.get('visual_elements', {})
            if visual_elements:
                elements = []
                if visual_elements.get('signatures', 0) > 0:
                    elements.append("potential signatures")
                if visual_elements.get('rectangles', 0) > 5:
                    elements.append("structured layout")
                if document_analysis.get('table_detected'):
                    elements.append("table data")
                
                if elements:
                    summary_parts.append(f"Contains: {', '.join(elements)}")
            
            return ". ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating enhanced summary: {str(e)}")
            return "Image processed successfully"