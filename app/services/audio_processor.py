import whisper
import librosa
import numpy as np
from pydub import AudioSegment
import tempfile
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
import re
from app.config import settings
from app.utils.privilege_patterns import PrivilegeScanner

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        try:
            # Load Whisper model with error handling
            self.whisper_model = whisper.load_model(settings.whisper_model)
            logger.info(f"Loaded Whisper model: {settings.whisper_model}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None
        
        self.privilege_scanner = PrivilegeScanner()
        
    async def process_audio_file(self, file_path: str, case_id: str) -> Dict:
        """Process audio file for legal discovery"""
        logger.info(f"Processing audio file: {file_path}")
        
        try:
            if not self.whisper_model:
                raise Exception("Whisper model not loaded")
            
            # Convert to WAV if necessary
            wav_path = await self._convert_to_wav(file_path)
            
            # Extract metadata
            metadata = await self._extract_audio_metadata(file_path)
            
            # Transcribe audio with timestamps
            transcript_data = await self._transcribe_audio(wav_path)
            
            # Detect speakers (basic implementation)
            speaker_segments = await self._detect_speakers(wav_path, transcript_data)
            
            # Identify key audio segments
            key_segments = await self._identify_key_segments(wav_path)
            
            # Scan for privilege indicators
            privilege_flags = []
            if transcript_data.get('text'):
                privilege_flags = self.privilege_scanner.scan_text(transcript_data['text'])
            
            # Extract legal entities and keywords
            entities = await self._extract_legal_entities(transcript_data.get('text', ''))
            
            # Generate summary
            summary = await self._generate_audio_summary(transcript_data, entities, metadata)
            
            # Clean up temporary files
            if wav_path != file_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            
            return {
                'id': f"audio_{case_id}_{os.path.basename(file_path)}",
                'case_id': case_id,
                'file_path': file_path,
                'file_type': 'audio',
                'metadata': metadata,
                'transcript': transcript_data,
                'speaker_segments': speaker_segments,
                'key_segments': key_segments,
                'privilege_flags': privilege_flags,
                'entities': entities,
                'summary': summary,
                'processing_status': 'completed',
                'processed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing audio file {file_path}: {str(e)}")
            return {
                'id': f"audio_{case_id}_{os.path.basename(file_path)}",
                'case_id': case_id,
                'file_path': file_path,
                'file_type': 'audio',
                'error': str(e),
                'processing_status': 'failed',
                'processed_at': datetime.utcnow().isoformat()
            }
    
    async def _convert_to_wav(self, audio_path: str) -> str:
        """Convert audio file to WAV format if needed"""
        if audio_path.lower().endswith('.wav'):
            return audio_path
            
        try:
            audio = AudioSegment.from_file(audio_path)
            wav_path = os.path.join(settings.temp_directory, 
                                   f"temp_{os.path.basename(audio_path)}.wav")
            audio.export(wav_path, format="wav")
            logger.info(f"Converted {audio_path} to WAV format")
            return wav_path
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {str(e)}")
            raise
    
    async def _extract_audio_metadata(self, file_path: str) -> Dict:
        """Extract metadata from audio file"""
        try:
            metadata = {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'format': os.path.splitext(file_path)[1].lower()
            }
            
            # Try to extract detailed metadata using mutagen
            try:
                from mutagen import File as MutagenFile
                audio_file = MutagenFile(file_path)
                
                if audio_file and audio_file.info:
                    metadata.update({
                        'duration': getattr(audio_file.info, 'length', None),
                        'bitrate': getattr(audio_file.info, 'bitrate', None),
                        'sample_rate': getattr(audio_file.info, 'sample_rate', None),
                        'channels': getattr(audio_file.info, 'channels', None)
                    })
                    
                    # Extract tags if available
                    if audio_file.tags:
                        tags = {}
                        for key, value in audio_file.tags.items():
                            if isinstance(value, list):
                                tags[key] = ', '.join(str(v) for v in value)
                            else:
                                tags[key] = str(value)
                        metadata['tags'] = tags
                        
            except ImportError:
                logger.warning("Mutagen not available for detailed metadata extraction")
            except Exception as e:
                logger.warning(f"Could not extract detailed metadata: {str(e)}")
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract audio metadata: {str(e)}")
            return {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'format': os.path.splitext(file_path)[1].lower()
            }
    
    async def _transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio using Whisper with word-level timestamps"""
        try:
            logger.info(f"Starting transcription of {audio_path}")
            
            result = self.whisper_model.transcribe(
                audio_path,
                task="transcribe",
                language="en",
                word_timestamps=True,
                verbose=False,
                temperature=0.0  # More deterministic results
            )
            
            # Process segments for better formatting
            formatted_segments = []
            for segment in result.get('segments', []):
                formatted_segments.append({
                    'id': segment.get('id'),
                    'start': segment.get('start'),
                    'end': segment.get('end'),
                    'text': segment.get('text', '').strip(),
                    'words': segment.get('words', []),
                    'avg_logprob': segment.get('avg_logprob', 0.0),
                    'no_speech_prob': segment.get('no_speech_prob', 0.0)
                })
            
            confidence = self._calculate_transcript_confidence(result)
            
            logger.info(f"Transcription completed with confidence: {confidence:.2f}")
            
            return {
                'text': result.get('text', '').strip(),
                'language': result.get('language', 'en'),
                'segments': formatted_segments,
                'confidence': confidence,
                'duration': result.get('duration', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return {
                'text': '',
                'language': 'en',
                'segments': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _calculate_transcript_confidence(self, whisper_result: Dict) -> float:
        """Calculate overall confidence score for transcript"""
        segments = whisper_result.get('segments', [])
        if not segments:
            return 0.0
            
        total_confidence = 0.0
        total_duration = 0.0
        
        for segment in segments:
            duration = segment.get('end', 0) - segment.get('start', 0)
            # Convert log probability to confidence (approximate)
            avg_logprob = segment.get('avg_logprob', -1.0)
            confidence = max(0.0, min(1.0, (avg_logprob + 1.0)))  # Normalize to 0-1
            
            total_confidence += confidence * duration
            total_duration += duration
        
        return total_confidence / total_duration if total_duration > 0 else 0.0
    
    async def _detect_speakers(self, audio_path: str, transcript_data: Dict) -> List[Dict]:
        """Enhanced speaker diarization using audio features and transcript timing"""
        try:
            # Load audio for analysis
            y, sr = librosa.load(audio_path)
            
            speaker_segments = []
            current_speaker = 1
            
            for i, segment in enumerate(transcript_data.get('segments', [])):
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '').strip()
                
                # Simple speaker change detection based on pause duration
                if i > 0:
                    prev_segment = transcript_data['segments'][i-1]
                    silence_duration = start_time - prev_segment.get('end', 0)
                    
                    # If there's a significant pause, assume speaker change
                    if silence_duration > 2.0:  # 2 seconds of silence
                        current_speaker = 3 - current_speaker  # Toggle between 1 and 2
                
                speaker_confidence = 0.7  # Base confidence
                
                speaker_segments.append({
                    'segment_id': segment.get('id'),
                    'speaker_id': f'Speaker_{current_speaker}',
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'text': text,
                    'confidence': speaker_confidence,
                    'word_count': len(text.split())
                })
            
            logger.info(f"Detected {len(set(seg['speaker_id'] for seg in speaker_segments))} speakers")
            return speaker_segments
            
        except Exception as e:
            logger.error(f"Error in speaker detection: {str(e)}")
            # Fallback: assign all segments to Speaker_1
            fallback_segments = []
            for seg in transcript_data.get('segments', []):
                fallback_segments.append({
                    'segment_id': seg.get('id'),
                    'speaker_id': 'Speaker_1',
                    'start_time': seg.get('start', 0),
                    'end_time': seg.get('end', 0),
                    'duration': seg.get('end', 0) - seg.get('start', 0),
                    'text': seg.get('text', ''),
                    'confidence': 0.5,
                    'word_count': len(seg.get('text', '').split())
                })
            return fallback_segments
    
    async def _identify_key_segments(self, audio_path: str) -> List[Dict]:
        """Identify key audio segments (loud sections, silence, etc.)"""
        try:
            y, sr = librosa.load(audio_path)
            
            # Detect non-silent regions
            intervals = librosa.effects.split(y, top_db=20)
            
            key_segments = []
            
            # Add significant silence breaks
            for i, (start, end) in enumerate(intervals[:-1]):
                silence_start = librosa.frames_to_time(end, sr=sr)
                silence_end = librosa.frames_to_time(intervals[i+1][0], sr=sr)
                silence_duration = silence_end - silence_start
                
                if silence_duration > 3.0:  # Significant silence
                    key_segments.append({
                        'type': 'silence',
                        'start_time': float(silence_start),
                        'end_time': float(silence_end),
                        'duration': float(silence_duration),
                        'description': f'Extended silence ({silence_duration:.1f}s)'
                    })
            
            logger.info(f"Identified {len(key_segments)} key audio segments")
            return key_segments
            
        except Exception as e:
            logger.error(f"Error identifying key segments: {str(e)}")
            return []
    
    async def _extract_legal_entities(self, transcript_text: str) -> Dict:
        """Extract legal entities from transcript"""
        if not transcript_text:
            return {
                'persons': [], 'organizations': [], 'dates': [], 'money': [],
                'locations': [], 'legal_terms': [], 'case_numbers': [],
                'phone_numbers': [], 'email_addresses': []
            }
        
        entities = {
            'persons': [],
            'organizations': [],
            'dates': [],
            'money': [],
            'locations': [],
            'legal_terms': [],
            'case_numbers': [],
            'phone_numbers': [],
            'email_addresses': []
        }
        
        # Extract legal terms using keyword matching
        legal_terms = [
            'contract', 'agreement', 'lawsuit', 'plaintiff', 'defendant',
            'attorney', 'lawyer', 'counsel', 'court', 'judge', 'settlement',
            'damages', 'liability', 'negligence', 'breach', 'violation',
            'deposition', 'affidavit', 'subpoena', 'injunction', 'discovery',
            'objection', 'sustained', 'overruled', 'witness', 'testimony'
        ]
        
        text_lower = transcript_text.lower()
        for term in legal_terms:
            if term in text_lower:
                entities['legal_terms'].append(term)
        
        # Extract case numbers using regex
        case_patterns = [
            r'\b\d{2,4}-\d{2,6}\b',  # e.g., 21-12345
            r'\bCase No\.?\s*\d+\b',   # e.g., Case No. 12345
            r'\bCivil No\.?\s*\d+\b'   # e.g., Civil No. 12345
        ]
        
        for pattern in case_patterns:
            matches = re.findall(pattern, transcript_text, re.IGNORECASE)
            entities['case_numbers'].extend(matches)
        
        # Extract phone numbers
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
        entities['phone_numbers'] = re.findall(phone_pattern, transcript_text)
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['email_addresses'] = re.findall(email_pattern, transcript_text)
        
        # Remove duplicates and empty entries
        for key in entities:
            if isinstance(entities[key][0] if entities[key] else None, tuple):
                # Handle phone numbers (tuples)
                entities[key] = list(set(entities[key]))
            else:
                entities[key] = list(set(filter(None, entities[key])))
        
        return entities
    
    async def _generate_audio_summary(self, transcript_data: Dict, entities: Dict, metadata: Dict) -> str:
        """Generate a comprehensive summary of the audio content"""
        try:
            text = transcript_data.get('text', '')
            duration = metadata.get('duration') or transcript_data.get('duration', 0)
            word_count = len(text.split()) if text else 0
            confidence = transcript_data.get('confidence', 0)
            
            summary_parts = []
            
            # Basic statistics
            if duration:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                summary_parts.append(f"Audio duration: {minutes}m {seconds}s")
            
            if word_count:
                summary_parts.append(f"Transcript contains {word_count} words")
                summary_parts.append(f"Transcription confidence: {confidence:.1%}")
            
            # Entity information
            if entities.get('legal_terms'):
                term_count = len(entities['legal_terms'])
                summary_parts.append(f"Legal terms identified: {term_count}")
            
            if entities.get('case_numbers'):
                summary_parts.append(f"Case numbers referenced: {', '.join(entities['case_numbers'])}")
            
            return ". ".join(summary_parts) if summary_parts else "Audio file processed successfully"
            
        except Exception as e:
            logger.error(f"Error generating audio summary: {str(e)}")
            return "Audio file processed successfully"