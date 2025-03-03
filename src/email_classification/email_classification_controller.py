from fastapi.params import Param
from .email_classification_service import EmailClassificationService
from .email_classification_model import EmailClassification
from nest.core import Controller, Get, Post, Injectable
from typing import Optional
from celery.result import AsyncResult
from utils.celery_config import celery_app
from utils.tasks import classify_email_batch
from src.database.mongodb import MongoDB
from fastapi import HTTPException, BackgroundTasks, Query
import logging


logger = logging.getLogger(__name__)

@Controller()
@Injectable()
class EmailClassificationController:
    def __init__(self, manual_email_classifier: EmailClassificationService,mongodb:MongoDB):
        self.classifier = manual_email_classifier
        logger = logging.Logger('EmailClassificationController')
        self.mongodb=mongodb

    @Post('/process-emails-async')
    async def process_emails_async(self, skip: int = Query(0), batch_size: int = Query(10)):
        """
        Endpoint to start asynchronous email processing.
        Returns a task ID that can be used to check the status.
        """
        try:
            # Start celery task
            task = classify_email_batch.delay(batch_size, skip)
            logger.info(f"Started async email processing task with ID: {task.id}")
            return {
                "status": "processing",
                "task_id": task.id,
                "message": "Email processing started"
            }
        except Exception as e:
            logger.error(f"Failed to start processing: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start processing: {str(e)}"
            )

    @Get('/task-status/:task_id')
    async def get_task_status(self, task_id: str = Param(None)):
        """
        Check the status of an async task
        """
        task_result = AsyncResult(task_id, app=celery_app)
        
        if task_result.ready():
            if task_result.successful():
                return {
                    "status": "completed",
                    "result": task_result.get()
                }
            else:
                return {
                    "status": "failed",
                    "error": str(task_result.result)
                }
        
        return {
            "status": "processing",
            "state": task_result.state
        }

    @Post('/process-email-batches')
    async def process_email_batches(self, 
                                  total_emails: int = Query(50), 
                                  batch_size: int = Query(10), 
                                  skip: int = Query(0)):
        """
        Process multiple batches of emails in parallel
        """
        try:
            logger.info(f"Processing emails with skip={skip}, batch_size={batch_size}")
            # Calculate number of batches
            num_batches = (total_emails + batch_size - 1) // batch_size
            task_ids = []
            
            # Start tasks for each batch
            original_skip = skip  
            for i in range(num_batches):
                current_skip = original_skip + (i * batch_size)
                task = classify_email_batch.delay(batch_size, current_skip)
                task_ids.append(task.id)
                
            return {
                "status": "processing",
                "task_ids": task_ids,
                "message": f"Started processing {num_batches} batches",
                "next_skip": original_skip + (num_batches * batch_size)
            }
            
        except Exception as e:
            logger.error(f"Failed to start batch processing: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start batch processing: {str(e)}"
            )

    @Get('/classification-results')
    async def get_classification_results(self,skip: int = 0, limit: int = 500):
        try:
            if self.mongodb.db is None:
                await self.mongodb.connect()
            logger.info("MongoDB connection established on demand")
            # Log the query parameters
            logger.info(f"Querying MongoDB with skip={skip}, limit={limit}")
            
            # Check collection existence and document count
            collection_names = await self.mongodb.db.list_collection_names()
            logger.info(f"Available collections: {collection_names}")
            
            total_docs = await self.mongodb.results.count_documents({})
            logger.info(f"Total documents in collection: {total_docs}")
            
            # Perform the query with detailed logging
            results = await self.mongodb.get_classification_results(skip=skip, limit=limit)
            logger.info(f"Retrieved {len(results)} results from MongoDB")
            
            # Log a sample document if available
            if results and len(results) > 0:
                logger.info(f"Sample document keys: {results[0].keys()}")
            
            # Transform MongoDB documents as needed
            transformed_results = []
            for result in results:
                # Add your transformation logic here
                transformed_results.append(result)
                
            logger.info(f"Transformed {len(transformed_results)} results for frontend")
            
            return {
                "status": "success",
                "results": transformed_results,
                "total": total_docs,
                "skip": skip,
                # "limit": limit
            }
        except Exception as e:
            logger.error(f"Error retrieving data from MongoDB: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )

    @Get('/results/category/:category')
    async def get_results_by_category(self, 
                                    category: str = Param(None), 
                                    skip: int = Query(0), 
                                    limit: int = Query(10)):
        """
        Get classification results filtered by category
        """
        try:
            results = await self.mongodb.get_results_by_category(category, skip, limit)
            
            # Transform MongoDB documents to match frontend format
            transformed_results = []
            for result in results:
                if 'classification_data' in result:
                    transformed_result = {
                        "email_details": result.get("classification_data", {}).get("email_details", {}),
                        "domain_analysis": result.get("classification_data", {}).get("domain_analysis", {}),
                        "initial_classification": result.get("classification_data", {}).get("initial_classification", {}),
                        "similar_emails": result.get("classification_data", {}).get("similar_emails", []),
                        "classification_process": result.get("classification_data", {}).get("classification_process", {}),
                        "status": result.get("status", "unknown")
                    }
                    transformed_results.append(transformed_result)
            
            return {
                "status": "success",
                "results": transformed_results,
                "category": category,
                "skip": skip,
                "limit": limit
            }
        except Exception as e:
            logger.error(f"Failed to fetch results by category: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch results by category: {str(e)}"
            )

    @Get('/health')
    async def health_check(self):
        """
        Simple health check endpoint
        """
        return {"status": "healthy"}