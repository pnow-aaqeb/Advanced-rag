# tasks.py
from src.celery.celery_config import celery_app
from prisma import Prisma
from openai import AsyncOpenAI
from .email_classification_service import EmailClassificationService
import os
import logging
import asyncio
from src.email_classification.pinecone_singleton_service import PineconeSingleton
from src.database.mongodb import MongoDB
from typing import Dict, Any

logger = logging.getLogger(__name__)

@celery_app.task(name='classify_email_batch')
def classify_email_batch(batch_size: int, skip: int) -> Dict[str, Any]:
    try:
        pinecone_index = PineconeSingleton().get_index()
        openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        async def process_batch():
            # Create all necessary dependencies
            prisma = Prisma()
            mongodb = MongoDB()  # Create MongoDB instance
            
            try:
                # Connect to services
                if not prisma.is_connected():
                    await prisma.connect()
                await mongodb.connect()  # Connect to MongoDB
                
                # Create the EmailClassificationService with required dependencies
                classifier = EmailClassificationService(mongodb=mongodb)  # Pass MongoDB
                
                # Set save_immediately attribute
                classifier.save_immediately = True
                
                # Get classification results
                result = await classifier.classify_emails(batch_size=batch_size, skip=skip, save_immediately=True)
                
                if not result:
                    logger.warning("No classification results returned")
                    return {
                        "status": "no_results",
                        "message": "No emails to classify"
                    }
                
                if result.get("status") == "error":
                    logger.error(f"Classification error: {result.get('message')}")
                    return result
                
                logger.info(f"Successfully processed batch. Status: {result.get('status')}")
                return result
                
            except Exception as e:
                logger.error(f"Error in process_batch: {str(e)}")
                raise
            finally:
                if prisma.is_connected():
                    await prisma.disconnect()

        # Run the async process
        result = asyncio.run(process_batch())
        
        if not result:
            logger.warning("No results returned from process_batch")
            return {
                "status": "error",
                "message": "No results returned from classification process"
            }
            
        logger.info(f"Successfully processed batch. Status: {result.get('status')}")
        return result
            
    except Exception as e:
        logger.error(f"Classification error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }