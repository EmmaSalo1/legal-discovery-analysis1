from celery import current_task
import logging
import asyncio
from app.workers.celery_app import celery_app
from app.services.video_processor import VideoProcessor

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def process_video_file_task(self, file_path: str, case_id: str):
    """Background task for processing video files"""
    try:
        self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting video processing'})
        
        processor = VideoProcessor()
        
        self.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Processing video'})
        
        # Process the file asynchronously
        result = asyncio.run(processor.process_video_file(file_path, case_id))
        
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Video processing complete'})
        
        return result
        
    except Exception as e:
        logger.error(f"Video processing task failed: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=3)