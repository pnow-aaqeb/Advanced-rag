# tasks.py
from celery_config import celery_app
from prisma import Prisma
from openai import AsyncOpenAI
from manual_flare import ManualEmailClassifier
import os
import logging
import asyncio
from pinecone_singleton import PineconeSingleton
from mongodb import mongodb
from typing import Dict, Any

logger = logging.getLogger(__name__)

@celery_app.task(name='classify_email_batch')
def classify_email_batch(batch_size: int, skip: int) -> Dict[str, Any]:
    try:
        # Initialize services
        pinecone_index = PineconeSingleton().get_index()
        openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        async def process_batch():
            prisma = Prisma()
            try:
                if not prisma.is_connected():
                    await prisma.connect()
                
                classifier = ManualEmailClassifier(
                    prisma_client=prisma,
                    pinecone_index=pinecone_index,
                    openai_client=openai_client
                )
                
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
                
                try:
                    # Save each result in the batch to MongoDB
                    saved_ids = []
                    if result.get("results"):
                        # Prepare batch result for MongoDB
                        mongo_result = {
                            "status": result.get("status"),
                            "results": result.get("results", []),
                            "total_processed": result.get("total_processed", 0),
                            "next_skip": result.get("next_skip")
                        }
                        
                        # Save to MongoDB
                        result_id = await mongodb.save_classification_result(mongo_result)
                        saved_ids.append(result_id)
                        
                        logger.info(f"Saved {len(saved_ids)} results to MongoDB")
                        
                        # Return success response with saved IDs
                        return {
                            "status": "success",
                            "result_ids": saved_ids,
                            "total_processed": len(saved_ids),
                            "message": "Classification completed and saved to database"
                        }
                    else:
                        return {
                            "status": "no_results",
                            "message": "No valid results to save"
                        }
                
                except Exception as mongo_error:
                    logger.error(f"MongoDB save error: {str(mongo_error)}")
                    # Return original results if MongoDB save fails
                    return {
                        "status": "error",
                        "message": f"Failed to save to database: {str(mongo_error)}",
                        "results": result.get("results", [])
                    }
                
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