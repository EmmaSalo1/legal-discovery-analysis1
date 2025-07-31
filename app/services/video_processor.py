import cv2
import os
import logging
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from app.services.audio_processor import EnhancedAudioProcessor
from app.services.image_processor import EnhancedImageProcessor
from app.config import settings

logger = logging.getLogger(__name__)

class EnhancedVideoProcessor:
    def __init__(self):
        self.audio_processor = EnhancedAudioProcessor()
        self.image_processor = EnhancedImageProcessor()
        
        # Check available backends
        self.backends = self._detect_available_backends()
        logger.info(f"Available video backends: {list(self.backends.keys())}")
        
    def _detect_available_backends(self) -> Dict[str, bool]:
        """Detect which video processing backends are available"""
        backends = {
            'opencv': True,  # Always available with opencv-python
            'ffmpeg': self._check_ffmpeg(),
            'moviepy': self._check_moviepy()
        }
        return backends
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _check_moviepy(self) -> bool:
        """Check if MoviePy is available"""
        try:
            import moviepy.editor as mp
            return True
        except ImportError:
            return False
    
    async def process_video_file(self, file_path: str, case_id: str) -> Dict:
        """Process video file with multiple backend support"""
        logger.info(f"Processing video file: {file_path}")
        
        try:
            # Extract basic metadata first
            metadata = await self._extract_video_metadata_opencv(file_path)
            
            if not metadata:
                raise ValueError("Could not read video file")
            
            # Check duration limits
            if metadata.get('duration', 0) > settings.max_video_duration:
                raise ValueError(f"Video duration exceeds maximum allowed ({settings.max_video_duration} seconds)")
            
            # Extract audio track using best available method
            audio_analysis = None
            audio_path = await self._extract_audio_track_best_method(file_path)
            
            if audio_path and os.path.exists(audio_path):
                try:
                    audio_analysis = await self.audio_processor.process_audio_file(audio_path, case_id)
                finally:
                    # Clean up temporary audio file
                    if os.path.exists(audio_path):
                        os.unlink(audio_path)
            
            # Extract key frames
            key_frames = await self._extract_key_frames_opencv(file_path)
            
            # Analyze visual content
            visual_analysis = await self._analyze_visual_content(file_path, key_frames)
            
            # Generate comprehensive summary
            summary = await self._generate_enhanced_summary(metadata, audio_analysis, visual_analysis, len(key_frames))
            
            return {
                'id': f"video_{case_id}_{os.path.basename(file_path)}",
                'case_id': case_id,
                'file_path': file_path,
                'file_type': 'video',
                'metadata': metadata,
                'audio_analysis': audio_analysis,
                'visual_analysis': visual_analysis,
                'key_frames_count': len(key_frames),
                'key_frames': key_frames[:10],  # Limit stored frames
                'summary': summary,
                'processing_status': 'completed',
                'processed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing video file {file_path}: {str(e)}")
            return {
                'id': f"video_{case_id}_{os.path.basename(file_path)}",
                'case_id': case_id,
                'file_path': file_path,
                'file_type': 'video',
                'error': str(e),
                'processing_status': 'failed',
                'processed_at': datetime.utcnow().isoformat()
            }
    
    async def _extract_video_metadata_opencv(self, file_path: str) -> Dict:
        """Extract video metadata using OpenCV"""
        try:
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video file: {file_path}")
                return {}
            
            # Get basic properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Try to get codec information
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            cap.release()
            
            metadata = {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'duration': duration,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'resolution': f"{width}x{height}",
                'aspect_ratio': round(width / height, 2) if height > 0 else 0,
                'codec': codec.strip('\x00'),
                'format': os.path.splitext(file_path)[1].lower(),
                'backend_used': 'opencv'
            }
            
            logger.info(f"Extracted metadata: {duration:.1f}s, {width}x{height}, {fps:.1f}fps")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting video metadata with OpenCV: {str(e)}")
            return {}
    
    async def _extract_audio_track_best_method(self, video_path: str) -> Optional[str]:
        """Extract audio using the best available method"""
        
        # Method 1: Try FFmpeg (best quality)
        if self.backends['ffmpeg']:
            audio_path = await self._extract_audio_ffmpeg(video_path)
            if audio_path:
                return audio_path
        
        # Method 2: Try MoviePy
        if self.backends['moviepy']:
            audio_path = await self._extract_audio_moviepy(video_path)
            if audio_path:
                return audio_path
        
        # Method 3: Try OpenCV + manual audio extraction
        return await self._extract_audio_opencv_fallback(video_path)
    
    async def _extract_audio_ffmpeg(self, video_path: str) -> Optional[str]:
        """Extract audio using FFmpeg"""
        try:
            temp_dir = settings.temp_directory
            audio_filename = f"temp_audio_{os.path.basename(video_path)}.wav"
            audio_path = os.path.join(temp_dir, audio_filename)
            
            # Use FFmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(audio_path):
                logger.info(f"Successfully extracted audio using FFmpeg: {audio_path}")
                return audio_path
            else:
                logger.warning(f"FFmpeg extraction failed: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"FFmpeg audio extraction failed: {str(e)}")
        
        return None
    
    async def _extract_audio_moviepy(self, video_path: str) -> Optional[str]:
        """Extract audio using MoviePy"""
        try:
            import moviepy.editor as mp
            
            temp_dir = settings.temp_directory
            audio_filename = f"temp_audio_{os.path.basename(video_path)}.wav"
            audio_path = os.path.join(temp_dir, audio_filename)
            
            video = mp.VideoFileClip(video_path)
            
            if video.audio is None:
                logger.info("No audio track found in video")
                video.close()
                return None
            
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            
            logger.info(f"Successfully extracted audio using MoviePy: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.warning(f"MoviePy audio extraction failed: {str(e)}")
        
        return None
    
    async def _extract_audio_opencv_fallback(self, video_path: str) -> Optional[str]:
        """Fallback method - inform user no audio extraction available"""
        logger.warning("No audio extraction method available. Install FFmpeg or MoviePy for audio processing.")
        return None
    
    async def _extract_key_frames_opencv(self, video_path: str) -> List[Dict]:
        """Extract key frames using OpenCV with intelligent frame selection"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            key_frames = []
            
            # Adaptive frame extraction based on video length
            if duration <= 60:  # Short video: every 10 seconds
                interval_seconds = 10
            elif duration <= 300:  # Medium video: every 30 seconds
                interval_seconds = 30
            else:  # Long video: every 60 seconds
                interval_seconds = 60
            
            frame_interval = int(fps * interval_seconds)
            max_frames = 20  # Limit total frames
            
            frame_num = 0
            extracted_count = 0
            
            while frame_num < frame_count and extracted_count < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                timestamp = frame_num / fps
                
                # Analyze frame content
                frame_analysis = await self._analyze_frame(frame, timestamp)
                
                key_frames.append({
                    'frame_number': frame_num,
                    'timestamp': timestamp,
                    'timestamp_formatted': f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                    'analysis': frame_analysis
                })
                
                frame_num += frame_interval
                extracted_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(key_frames)} key frames")
            return key_frames
            
        except Exception as e:
            logger.error(f"Error extracting key frames: {str(e)}")
            return []
    
    async def _analyze_frame(self, frame: np.ndarray, timestamp: float) -> Dict:
        """Analyze individual frame content"""
        try:
            analysis = {
                'timestamp': timestamp,
                'brightness': 0,
                'motion_score': 0,
                'text_detected': False,
                'face_count': 0,
                'dominant_colors': []
            }
            
            # Calculate brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            analysis['brightness'] = float(np.mean(gray))
            
            # Detect faces (basic)
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                analysis['face_count'] = len(faces)
            except:
                pass
            
            # Basic text detection using contours
            try:
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Simple heuristic: many small rectangular contours might indicate text
                text_like_contours = 0
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.2 < aspect_ratio < 5 and 10 < w < 200 and 5 < h < 50:
                        text_like_contours += 1
                
                analysis['text_detected'] = text_like_contours > 10
            except:
                pass
            
            # Dominant colors
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pixels = frame_rgb.reshape(-1, 3)
                
                # Sample pixels for performance
                sample_size = min(1000, len(pixels))
                sampled_pixels = pixels[::len(pixels)//sample_size]
                
                # Find dominant colors using k-means
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                kmeans.fit(sampled_pixels)
                
                dominant_colors = []
                for color in kmeans.cluster_centers_:
                    dominant_colors.append([int(c) for c in color])
                
                analysis['dominant_colors'] = dominant_colors
            except:
                pass
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Frame analysis error: {str(e)}")
            return {'timestamp': timestamp}
    
    async def _analyze_visual_content(self, video_path: str, key_frames: List[Dict]) -> Dict:
        """Analyze visual content across the video"""
        try:
            analysis = {
                'scene_changes': 0,
                'average_brightness': 0,
                'has_faces': False,
                'has_text': False,
                'visual_complexity': 'medium',
                'dominant_color_palette': [],
                'motion_level': 'medium'
            }
            
            if not key_frames:
                return analysis
            
            # Analyze brightness trends
            brightness_values = [frame.get('analysis', {}).get('brightness', 0) for frame in key_frames]
            if brightness_values:
                analysis['average_brightness'] = sum(brightness_values) / len(brightness_values)
            
            # Check for faces and text
            face_frames = sum(1 for frame in key_frames if frame.get('analysis', {}).get('face_count', 0) > 0)
            text_frames = sum(1 for frame in key_frames if frame.get('analysis', {}).get('text_detected', False))
            
            analysis['has_faces'] = face_frames > 0
            analysis['has_text'] = text_frames > 0
            
            # Estimate scene changes based on brightness variations
            if len(brightness_values) > 1:
                brightness_changes = sum(1 for i in range(1, len(brightness_values)) 
                                       if abs(brightness_values[i] - brightness_values[i-1]) > 30)
                analysis['scene_changes'] = brightness_changes
            
            # Collect dominant colors
            all_colors = []
            for frame in key_frames:
                colors = frame.get('analysis', {}).get('dominant_colors', [])
                all_colors.extend(colors)
            
            if all_colors:
                # Simple color clustering
                analysis['dominant_color_palette'] = all_colors[:6]  # Top 6 colors
            
            logger.info(f"Visual analysis complete: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"Visual content analysis error: {str(e)}")
            return {}
    
    async def _generate_enhanced_summary(self, metadata: Dict, audio_analysis: Optional[Dict], 
                                       visual_analysis: Dict, frame_count: int) -> str:
        """Generate comprehensive video summary"""
        try:
            summary_parts = []
            
            # Basic info
            duration = metadata.get('duration', 0)
            if duration:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                summary_parts.append(f"Video duration: {minutes}m {seconds}s")
            
            summary_parts.append(f"Resolution: {metadata.get('resolution', 'Unknown')}")
            summary_parts.append(f"Frame rate: {metadata.get('fps', 0):.1f} FPS")
            
            # Audio content
            if audio_analysis and 'transcript' in audio_analysis:
                transcript_text = audio_analysis['transcript'].get('text', '')
                if transcript_text:
                    word_count = len(transcript_text.split())
                    confidence = audio_analysis['transcript'].get('confidence', 0)
                    summary_parts.append(f"Audio transcript: {word_count} words ({confidence:.1%} confidence)")
                else:
                    summary_parts.append("Audio track present but no speech detected")
            else:
                summary_parts.append("No audio track or audio processing failed")
            
            # Visual content
            if visual_analysis:
                if visual_analysis.get('has_faces'):
                    summary_parts.append("Contains people/faces")
                
                if visual_analysis.get('has_text'):
                    summary_parts.append("Contains visible text")
                
                scene_changes = visual_analysis.get('scene_changes', 0)
                if scene_changes > 0:
                    summary_parts.append(f"Detected {scene_changes} scene changes")
                
                brightness = visual_analysis.get('average_brightness', 0)
                if brightness > 0:
                    brightness_desc = "bright" if brightness > 150 else "normal" if brightness > 80 else "dark"
                    summary_parts.append(f"Video lighting: {brightness_desc}")
            
            if frame_count > 0:
                summary_parts.append(f"Extracted {frame_count} key frames for analysis")
            
            return ". ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating video summary: {str(e)}")
            return "Video processed successfully"