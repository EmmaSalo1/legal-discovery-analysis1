import cv2
import moviepy.editor as mp
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
from app.services.audio_processor import AudioProcessor
from app.services.image_processor import ImageProcessor
from app.config import settings

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.image_processor = ImageProcessor()
        
    async def process_video_file(self, file_path: str, case_id: str) -> Dict:
        """Process video file for legal discovery"""
        logger.info(f"Processing video file: {file_path}")
        
        try:
            # Extract metadata
            metadata = await self._extract_video_metadata(file_path)
            
            # Check duration limits
            if metadata.get('duration', 0) > settings.max_video_duration:
                raise ValueError(f"Video duration exceeds maximum allowed ({settings.max_video_duration} seconds)")
            
            # Extract audio track
            audio_path = await self._extract_audio_track(file_path)
            audio_analysis = None
            
            if audio_path and os.path.exists(audio_path):
                audio_analysis = await self.audio_processor.process_audio_file(audio_path, case_id)
                # Clean up temporary audio file
                os.unlink(audio_path)
            
            # Extract key frames (simplified for Mac compatibility)
            key_frames = await self._extract_key_frames(file_path)
            
            # Generate summary
            summary = await self._generate_summary(metadata, audio_analysis, len(key_frames))
            
            return {
                'id': f"video_{case_id}_{os.path.basename(file_path)}",
                'case_id': case_id,
                'file_path': file_path,
                'file_type': 'video',
                'metadata': metadata,
                'audio_analysis': audio_analysis,
                'key_frames_count': len(key_frames),
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
    
    async def _extract_video_metadata(self, file_path: str) -> Dict:
        """Extract video metadata"""
        try:
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'duration': duration,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'resolution': f"{width}x{height}",
                'format': os.path.splitext(file_path)[1].lower()
            }
            
        except Exception as e:
            logger.error(f"Error extracting video metadata: {str(e)}")
            return {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'format': os.path.splitext(file_path)[1].lower()
            }
    
    async def _extract_audio_track(self, video_path: str) -> Optional[str]:
        """Extract audio track from video"""
        try:
            video = mp.VideoFileClip(video_path)
            
            if video.audio is None:
                logger.info("No audio track found in video")
                video.close()
                return None
            
            audio_path = os.path.join(settings.temp_directory, 
                                     f"temp_audio_{os.path.basename(video_path)}.wav")
            
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Error extracting audio track: {str(e)}")
            return None
    
    async def _extract_key_frames(self, video_path: str) -> List[Dict]:
        """Extract key frames (simplified version)"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            key_frames = []
            # Extract one frame every 60 seconds for simplicity
            frame_interval = 60 * fps
            
            frame_num = 0
            while frame_num < frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                timestamp = frame_num / fps
                key_frames.append({
                    'frame_number': frame_num,
                    'timestamp': timestamp
                })
                
                frame_num += int(frame_interval)
                if len(key_frames) >= 10:  # Limit to 10 frames
                    break
            
            cap.release()
            return key_frames
            
        except Exception as e:
            logger.error(f"Error extracting key frames: {str(e)}")
            return []
    
    async def _generate_summary(self, metadata: Dict, audio_analysis: Optional[Dict], frame_count: int) -> str:
        """Generate video summary"""
        try:
            summary_parts = [
                f"Video duration: {metadata.get('duration', 0):.1f} seconds",
                f"Resolution: {metadata.get('resolution', 'Unknown')}",
                f"Frame rate: {metadata.get('fps', 0):.1f} FPS"
            ]
            
            if frame_count > 0:
                summary_parts.append(f"Key frames extracted: {frame_count}")
            
            if audio_analysis and 'transcript' in audio_analysis:
                transcript_text = audio_analysis['transcript'].get('text', '')
                if transcript_text:
                    word_count = len(transcript_text.split())
                    summary_parts.append(f"Audio transcript: {word_count} words")
            
            return ". ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating video summary: {str(e)}")
            return "Video processed successfully"
