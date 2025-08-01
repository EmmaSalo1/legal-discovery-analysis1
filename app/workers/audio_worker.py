from celery import current_task
import logging
import asyncio
from app.workers.celery_app import celery_app
from app.services.audio_processor import EnhancedAudioProcessor

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def process_audio_file_task(self, file_path: str, case_id: str):
    """Background task for processing audio files"""
    try:
        self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting audio processing'})
        
        processor = EnhancedAudioProcessor()
        
        self.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Transcribing audio'})
        
        # Process the file asynchronously
        result = asyncio.run(processor.process_audio_file(file_path, case_id))
        
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Audio processing complete'})
        
        return result
        
    except Exception as e:
        logger.error(f"Audio processing task failed: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=3)