# from celery import Celery
# from kombu import Exchange,Queue

# celery_app=Celery('email_embeddings', broker='redis://localhost:6380/0',backend='redis://localhost:6380/1')
# main_exchange=Exchange('main',type='direct')

# celery_app.conf.task_queues=(

#     Queue('queue_1', main_exchange, routing_key='queue_1'),
    
#     Queue('queue_2', main_exchange, routing_key='queue_2'),
# )
# celery_app.conf.imports = ['embedding_generation']
# celery_app.conf.update(
#     task_serializer='json',
#     accept_content=['json'],
#     result_serializer='json',
#     timezone='UTC',
#     enable_utc=True,
#     task_acks_late=True,
#     worker_prefetch_multiplier=2,
#     task_time_limit=3600
# )
# celery_config.py
# celery_config.py
from celery import Celery
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

load_dotenv()

# Initialize Celery with Redis as broker
celery_app = Celery(
    'email_classifier',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6380/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6380/0')
)

# Configure Celery settings
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour timeout
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks
    worker_prefetch_multiplier=4,  # Number of tasks each worker prefetches
    broker_connection_retry_on_startup=True,
    worker_redirect_stdouts=False,
    worker_redirect_stdouts_level='INFO'
)