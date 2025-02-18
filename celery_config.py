from celery import Celery
from kombu import Exchange,Queue

celery_app=Celery('email_embeddings', broker='redis://localhost:6380/0',backend='redis://localhost:6380/1')
main_exchange=Exchange('main',type='direct')

celery_app.conf.task_queues=(

    Queue('queue_1', main_exchange, routing_key='queue_1'),
    
    Queue('queue_2', main_exchange, routing_key='queue_2'),
)
celery_app.conf.imports = ['embedding_generation']
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=3,
    task_time_limit=3600
)
