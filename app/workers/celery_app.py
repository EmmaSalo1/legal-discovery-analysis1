from celery import Celery
from app.config import settings

# Create Celery app
celery_app = Celery(
    "legal_discovery",
    broker=settings.celery_broker_url,
    backend=settings.celery_broker_url,
    include=["app.workers.audio_worker", "app.workers.video_worker", "app.workers.image_worker"]
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3300,  # 55 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)