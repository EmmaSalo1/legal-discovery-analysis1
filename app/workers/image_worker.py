from celery import current_task
import logging
from app.workers.celery_app import celery_app
from app.services.image_processor import ImageProcessor

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def process_image_file_task(self, file_path: str, case_id: str):
    """Background task for processing image files"""
    try:
        self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting image processing'})
        
        processor = ImageProcessor()
        
        self.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Performing OCR'})
        
        result = processor.process_image_file(file_path, case_id)
        
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Image processing complete'})
        
        return result
        
    except Exception as e:
        logger.error(f"Image processing task failed: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=3)