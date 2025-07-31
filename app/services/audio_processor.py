import whisper
import librosa
import numpy as np
from pydub import AudioSegment
import tempfile
import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
import json
from app.config import settings
from app.utils.privilege_patterns import PrivilegeScanner

logger = logging.getLogger(__name__)

class EnhancedAudioProcessor:
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
        """Enhanced audio processing with detailed analysis"""
        logger.info(f"Processing audio file: {file_path}")
        
        try:
            if not self.whisper_model:
                raise Exception("Whisper model not loaded")
            
            # Convert to WAV if necessary
            wav_path = await self._convert_to_wav(file_path)
            
            # Extract comprehensive metadata
            metadata = await self._extract_audio_metadata(file_path)
            
            # Transcribe audio with enhanced analysis
            transcript_data = await self._transcribe_audio_enhanced(wav_path)
            
            # Advanced speaker analysis
            speaker_analysis = await self._advanced_speaker_analysis(wav_path, transcript_data)
            
            # Audio quality and content analysis
            audio_analysis = await self._analyze_audio_quality(wav_path)
            
            # Extract legal and factual information
            content_analysis = await self._analyze_audio_content(transcript_data.get('text', ''))
            
            # Scan for privilege indicators
            privilege_flags = []
            if transcript_data.get('text'):
                privilege_flags = self.privilege_scanner.scan_text(transcript_data['text'])
            
            # Generate comprehensive summary
            summary = await self._generate_audio_summary(
                transcript_data, speaker_analysis, audio_analysis, content_analysis, metadata
            )
            
            # Clean up temporary files
            if wav_path != file_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            
            result = {
                'id': f"audio_{case_id}_{os.path.basename(file_path)}",
                'case_id': case_id,
                'file_path': file_path,
                'file_type': 'audio',
                'metadata': metadata,
                'transcript': transcript_data,
                'speaker_analysis': speaker_analysis,
                'audio_analysis': audio_analysis,
                'content_analysis': content_analysis,
                'privilege_flags': privilege_flags,
                'summary': summary,
                'processing_status': 'completed',
                'processed_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Successfully processed audio file: {file_path}")
            return result
            
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
    
    async def _transcribe_audio_enhanced(self, audio_path: str) -> Dict:
        """Enhanced transcription with detailed timing and confidence"""
        try:
            logger.info(f"Starting enhanced transcription of {audio_path}")
            
            # Use Whisper with word-level timestamps and higher precision
            result = self.whisper_model.transcribe(
                audio_path,
                task="transcribe",
                language="en",
                word_timestamps=True,
                verbose=False,
                temperature=0.0,
                best_of=3,  # Try multiple attempts for better accuracy
                beam_size=5  # Better search for optimal transcription
            )
            
            # Enhanced segment processing
            enhanced_segments = []
            for segment in result.get('segments', []):
                enhanced_segment = {
                    'id': segment.get('id'),
                    'start': segment.get('start'),
                    'end': segment.get('end'),
                    'duration': segment.get('end', 0) - segment.get('start', 0),
                    'text': segment.get('text', '').strip(),
                    'words': segment.get('words', []),
                    'avg_logprob': segment.get('avg_logprob', 0.0),
                    'no_speech_prob': segment.get('no_speech_prob', 0.0),
                    'confidence': self._segment_confidence(segment),
                    'word_count': len(segment.get('text', '').split()),
                    'speech_rate': self._calculate_speech_rate(segment)
                }
                enhanced_segments.append(enhanced_segment)
            
            # Overall confidence and quality metrics
            overall_confidence = self._calculate_transcript_confidence(result)
            quality_metrics = self._assess_transcript_quality(enhanced_segments)
            
            # Generate speaker-aware transcript
            formatted_transcript = self._format_transcript_with_timing(enhanced_segments)
            
            transcript_result = {
                'text': result.get('text', '').strip(),
                'language': result.get('language', 'en'),
                'segments': enhanced_segments,
                'confidence': overall_confidence,
                'duration': result.get('duration', 0.0),
                'quality_metrics': quality_metrics,
                'formatted_transcript': formatted_transcript,
                'word_count': len(result.get('text', '').split()),
                'processing_info': {
                    'model': settings.whisper_model,
                    'timestamp_precision': 'word-level'
                }
            }
            
            logger.info(f"Transcription completed with {overall_confidence:.2f} confidence")
            return transcript_result
            
        except Exception as e:
            logger.error(f"Error in enhanced transcription: {str(e)}")
            return {
                'text': '',
                'language': 'en',
                'segments': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _segment_confidence(self, segment: Dict) -> float:
        """Calculate confidence score for a segment"""
        try:
            avg_logprob = segment.get('avg_logprob', -1.0)
            no_speech_prob = segment.get('no_speech_prob', 1.0)
            
            # Convert log probability to confidence
            logprob_confidence = max(0.0, min(1.0, (avg_logprob + 1.0)))
            
            # Factor in no-speech probability
            speech_confidence = 1.0 - no_speech_prob
            
            # Combined confidence
            return (logprob_confidence + speech_confidence) / 2.0
            
        except:
            return 0.5
    
    def _calculate_speech_rate(self, segment: Dict) -> float:
        """Calculate words per minute for a segment"""
        try:
            duration = segment.get('end', 0) - segment.get('start', 0)
            word_count = len(segment.get('text', '').split())
            
            if duration > 0:
                return (word_count / duration) * 60  # Words per minute
            return 0.0
        except:
            return 0.0
    
    def _assess_transcript_quality(self, segments: List[Dict]) -> Dict:
        """Assess overall transcript quality"""
        try:
            if not segments:
                return {'overall_quality': 'poor', 'issues': ['no_segments']}
            
            # Calculate various quality metrics
            avg_confidence = sum(seg.get('confidence', 0) for seg in segments) / len(segments)
            avg_speech_rate = sum(seg.get('speech_rate', 0) for seg in segments) / len(segments)
            
            # Detect potential issues
            issues = []
            if avg_confidence < 0.5:
                issues.append('low_confidence')
            if avg_speech_rate > 200:
                issues.append('very_fast_speech')
            elif avg_speech_rate < 80:
                issues.append('very_slow_speech')
            
            # Check for very short segments (might indicate poor audio)
            short_segments = sum(1 for seg in segments if seg.get('duration', 0) < 1.0)
            if short_segments / len(segments) > 0.5:
                issues.append('fragmented_speech')
            
            # Overall quality assessment
            if avg_confidence > 0.8 and not issues:
                overall_quality = 'excellent'
            elif avg_confidence > 0.6 and len(issues) <= 1:
                overall_quality = 'good'
            elif avg_confidence > 0.4:
                overall_quality = 'fair'
            else:
                overall_quality = 'poor'
            
            return {
                'overall_quality': overall_quality,
                'average_confidence': avg_confidence,
                'average_speech_rate': avg_speech_rate,
                'issues': issues,
                'total_segments': len(segments)
            }
            
        except Exception as e:
            logger.warning(f"Quality assessment error: {e}")
            return {'overall_quality': 'unknown', 'error': str(e)}
    
    def _format_transcript_with_timing(self, segments: List[Dict]) -> str:
        """Format transcript with timestamps for easy reading"""
        try:
            formatted_lines = []
            
            for segment in segments:
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '').strip()
                
                if text:
                    # Format time as MM:SS
                    start_formatted = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
                    end_formatted = f"{int(end_time//60):02d}:{int(end_time%60):02d}"
                    
                    formatted_lines.append(f"[{start_formatted}-{end_formatted}] {text}")
            
            return "\n".join(formatted_lines)
            
        except Exception as e:
            logger.warning(f"Transcript formatting error: {e}")
            return ""
    
    async def _advanced_speaker_analysis(self, audio_path: str, transcript_data: Dict) -> Dict:
        """Advanced speaker diarization and analysis"""
        try:
            # Load audio for analysis
            y, sr = librosa.load(audio_path)
            
            speaker_segments = []
            segments = transcript_data.get('segments', [])
            
            if not segments:
                return {'speakers': [], 'speaker_count': 0}
            
            # Enhanced speaker change detection
            current_speaker = 1
            speaker_changes = []
            
            for i, segment in enumerate(segments):
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '').strip()
                confidence = segment.get('confidence', 0.5)
                
                # Detect speaker changes based on multiple factors
                speaker_changed = False
                
                if i > 0:
                    prev_segment = segments[i-1]
                    silence_duration = start_time - prev_segment.get('end', 0)
                    
                    # Factors indicating speaker change
                    if silence_duration > 2.0:  # Long pause
                        speaker_changed = True
                    elif silence_duration > 1.0 and self._text_suggests_speaker_change(text, prev_segment.get('text', '')):
                        speaker_changed = True
                
                if speaker_changed:
                    current_speaker = 3 - current_speaker  # Toggle between 1 and 2
                    speaker_changes.append({
                        'time': start_time,
                        'reason': 'pause_and_context' if silence_duration > 1.0 else 'long_pause'
                    })
                
                # Estimate speaker confidence based on audio features
                speaker_confidence = min(0.9, 0.5 + confidence * 0.4)
                
                speaker_segments.append({
                    'segment_id': segment.get('id'),
                    'speaker_id': f'Speaker_{current_speaker}',
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'text': text,
                    'confidence': speaker_confidence,
                    'word_count': len(text.split()),
                    'speech_rate': segment.get('speech_rate', 0)
                })
            
            # Analyze speaker characteristics
            speaker_stats = self._analyze_speaker_characteristics(speaker_segments)
            
            return {
                'speaker_segments': speaker_segments,
                'speaker_changes': speaker_changes,
                'speaker_count': len(set(seg['speaker_id'] for seg in speaker_segments)),
                'speaker_statistics': speaker_stats
            }
            
        except Exception as e:
            logger.error(f"Error in speaker analysis: {str(e)}")
            return {'speakers': [], 'speaker_count': 0, 'error': str(e)}
    
    def _text_suggests_speaker_change(self, current_text: str, previous_text: str) -> bool:
        """Analyze text content to suggest speaker changes"""
        try:
            # Look for conversational patterns
            speaker_indicators = [
                r'\b(yes|yeah|okay|right|mm-hmm|uh-huh)\b',  # Agreements
                r'\b(no|nope|never|don\'t|can\'t)\b',        # Disagreements
                r'\b(well|so|actually|but|however)\b',       # Discourse markers
                r'\b(thank you|thanks|please|excuse me)\b'    # Politeness markers
            ]
            
            # Question-answer patterns
            if ('?' in previous_text and 
                any(word in current_text.lower() for word in ['yes', 'no', 'well', 'i'])):
                return True
            
            # Check for sudden topic changes
            prev_words = set(previous_text.lower().split())
            curr_words = set(current_text.lower().split())
            overlap = len(prev_words & curr_words) / max(len(prev_words), 1)
            
            if overlap < 0.2:  # Very different content
                return True
                
            return False
            
        except:
            return False
    
    def _analyze_speaker_characteristics(self, speaker_segments: List[Dict]) -> Dict:
        """Analyze characteristics of each speaker"""
        try:
            speakers = {}
            
            for segment in speaker_segments:
                speaker_id = segment['speaker_id']
                
                if speaker_id not in speakers:
                    speakers[speaker_id] = {
                        'total_time': 0,
                        'total_words': 0,
                        'segments': 0,
                        'speech_rates': [],
                        'avg_confidence': 0
                    }
                
                speakers[speaker_id]['total_time'] += segment['duration']
                speakers[speaker_id]['total_words'] += segment['word_count']
                speakers[speaker_id]['segments'] += 1
                speakers[speaker_id]['speech_rates'].append(segment.get('speech_rate', 0))
                speakers[speaker_id]['avg_confidence'] += segment['confidence']
            
            # Calculate averages and characteristics
            for speaker_id, stats in speakers.items():
                if stats['segments'] > 0:
                    stats['avg_confidence'] /= stats['segments']
                    stats['avg_speech_rate'] = sum(stats['speech_rates']) / len(stats['speech_rates'])
                    stats['participation_percentage'] = (stats['total_time'] / 
                        sum(s['total_time'] for s in speakers.values()) * 100)
                    
                    # Characterize speaking style
                    if stats['avg_speech_rate'] > 160:
                        stats['speaking_style'] = 'fast'
                    elif stats['avg_speech_rate'] < 100:
                        stats['speaking_style'] = 'slow'
                    else:
                        stats['speaking_style'] = 'normal'
                
                # Clean up intermediate data
                del stats['speech_rates']
            
            return speakers
            
        except Exception as e:
            logger.warning(f"Speaker characteristics analysis error: {e}")
            return {}
    
    async def _analyze_audio_quality(self, audio_path: str) -> Dict:
        """Analyze audio quality and characteristics"""
        try:
            y, sr = librosa.load(audio_path)
            
            analysis = {
                'sample_rate': sr,
                'duration': len(y) / sr,
                'channels': 1,  # librosa loads as mono by default
                'dynamic_range': 0,
                'noise_level': 0,
                'clarity_score': 0.5,
                'volume_consistency': 0.5
            }
            
            # Dynamic range analysis
            rms = librosa.feature.rms(y=y)[0]
            analysis['dynamic_range'] = float(np.max(rms) - np.min(rms))
            
            # Estimate noise level
            # Use the quietest 10% of the audio as noise baseline
            sorted_rms = np.sort(rms)
            noise_threshold_idx = int(len(sorted_rms) * 0.1)
            analysis['noise_level'] = float(np.mean(sorted_rms[:noise_threshold_idx]))
            
            # Volume consistency
            volume_std = np.std(rms)
            analysis['volume_consistency'] = max(0, 1 - (volume_std / np.mean(rms)))
            
            # Simple clarity estimate based on spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # Higher spectral centroid often indicates clearer speech
            avg_centroid = np.mean(spectral_centroids)
            analysis['clarity_score'] = min(1.0, avg_centroid / 2000)  # Normalize
            
            # Overall quality assessment
            quality_factors = [
                analysis['clarity_score'],
                analysis['volume_consistency'],
                1 - min(1.0, analysis['noise_level'] * 10)  # Lower noise = better
            ]
            
            overall_quality = sum(quality_factors) / len(quality_factors)
            
            if overall_quality > 0.8:
                analysis['quality_assessment'] = 'excellent'
            elif overall_quality > 0.6:
                analysis['quality_assessment'] = 'good'
            elif overall_quality > 0.4:
                analysis['quality_assessment'] = 'fair'
            else:
                analysis['quality_assessment'] = 'poor'
            
            analysis['overall_quality_score'] = overall_quality
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Audio quality analysis error: {e}")
            return {'quality_assessment': 'unknown', 'error': str(e)}
    
    async def _analyze_audio_content(self, transcript_text: str) -> Dict:
        """Analyze the content of the audio transcript for legal relevance"""
        try:
            if not transcript_text:
                return {'content_type': 'unknown', 'topics': [], 'legal_relevance': 'low'}
            
            analysis = {
                'content_type': 'conversation',
                'topics': [],
                'legal_terms': [],
                'temporal_references': [],
                'people_mentioned': [],
                'locations_mentioned': [],
                'financial_references': [],
                'emotional_indicators': [],
                'legal_relevance': 'medium'
            }
            
            text_lower = transcript_text.lower()
            
            # Detect content type
            if any(word in text_lower for word in ['deposition', 'testimony', 'oath', 'court']):
                analysis['content_type'] = 'legal_proceeding'
            elif any(word in text_lower for word in ['meeting', 'conference', 'discussion']):
                analysis['content_type'] = 'meeting'
            elif any(word in text_lower for word in ['phone', 'call', 'calling']):
                analysis['content_type'] = 'phone_call'
            elif any(word in text_lower for word in ['interview', 'question', 'answer']):
                analysis['content_type'] = 'interview'
            
            # Extract legal terms
            legal_terms = [
                'contract', 'agreement', 'lawsuit', 'plaintiff', 'defendant',
                'attorney', 'lawyer', 'counsel', 'court', 'judge', 'settlement',
                'damages', 'liability', 'negligence', 'breach', 'violation',
                'evidence', 'witness', 'testimony', 'objection', 'sustained',
                'overruled', 'guilty', 'innocent', 'verdict', 'appeal'
            ]
            
            found_legal_terms = [term for term in legal_terms if term in text_lower]
            analysis['legal_terms'] = found_legal_terms
            
            # Extract temporal references
            temporal_patterns = [
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
                r'\b(yesterday|today|tomorrow|last week|next week|last month|next month)\b',
                r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
            ]
            
            for pattern in temporal_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                analysis['temporal_references'].extend(matches)
            
            # Extract financial references
            financial_patterns = [
                r'\$[\d,]+',
                r'\b\d+\s*dollars?\b',
                r'\b(payment|money|cash|check|invoice|bill|fee|cost|price)\b'
            ]
            
            for pattern in financial_patterns:
                matches = re.findall(pattern, transcript_text, re.IGNORECASE)
                analysis['financial_references'].extend(matches)
            
            # Detect emotional indicators
            emotional_words = [
                'angry', 'upset', 'frustrated', 'happy', 'sad', 'worried',
                'concerned', 'excited', 'nervous', 'calm', 'stressed'
            ]
            
            found_emotions = [word for word in emotional_words if word in text_lower]
            analysis['emotional_indicators'] = found_emotions
            
            # Assess legal relevance
            relevance_score = 0
            if found_legal_terms:
                relevance_score += len(found_legal_terms) * 0.2
            if analysis['financial_references']:
                relevance_score += 0.3
            if analysis['content_type'] in ['legal_proceeding', 'deposition']:
                relevance_score += 0.5
            
            if relevance_score > 0.8:
                analysis['legal_relevance'] = 'high'
            elif relevance_score > 0.4:
                analysis['legal_relevance'] = 'medium'
            else:
                analysis['legal_relevance'] = 'low'
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Content analysis error: {e}")
            return {'content_type': 'unknown', 'error': str(e)}
    
    async def _generate_audio_summary(self, transcript_data: Dict, speaker_analysis: Dict, 
                                    audio_analysis: Dict, content_analysis: Dict, metadata: Dict) -> str:
        """Generate comprehensive audio summary"""
        try:
            summary_parts = []
            
            # Basic information
            duration = metadata.get('duration') or transcript_data.get('duration', 0)
            if duration:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                summary_parts.append(f"Audio duration: {minutes}m {seconds}s")
            
            # Transcript quality
            confidence = transcript_data.get('confidence', 0)
            word_count = transcript_data.get('word_count', 0)
            
            if word_count > 0:
                summary_parts.append(f"Transcript: {word_count} words ({confidence:.1%} confidence)")
            
            # Quality assessment
            quality = audio_analysis.get('quality_assessment', 'unknown')
            if quality != 'unknown':
                summary_parts.append(f"Audio quality: {quality}")
            
            # Speaker information
            speaker_count = speaker_analysis.get('speaker_count', 0)
            if speaker_count > 0:
                summary_parts.append(f"Speakers identified: {speaker_count}")
            
            # Content type and relevance
            content_type = content_analysis.get('content_type', 'conversation')
            legal_relevance = content_analysis.get('legal_relevance', 'medium')
            summary_parts.append(f"Content type: {content_type}")
            
            if legal_relevance == 'high':
                summary_parts.append("High legal relevance detected")
            
            # Legal terms
            legal_terms = content_analysis.get('legal_terms', [])
            if legal_terms:
                summary_parts.append(f"Legal terms: {', '.join(legal_terms[:3])}")
            
            # Financial references
            financial_refs = content_analysis.get('financial_references', [])
            if financial_refs:
                summary_parts.append(f"Financial references: {len(financial_refs)} found")
            
            return ". ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating audio summary: {str(e)}")
            return "Audio file processed successfully"