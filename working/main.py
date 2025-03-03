from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from prisma import Prisma
from pinecone import Pinecone
import logging
from manual_flare import ManualEmailClassifier
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from celery.result import AsyncResult
from utils.celery_config import celery_app
from utils.tasks import classify_email_batch

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize connections
# prisma = Prisma()
# pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
# openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Define the results file path
RESULTS_FILE = "email_classification_results.json"

async def load_existing_results():
    """
    Load existing results from the JSON file.
    If the file doesn't exist or is empty, return an empty list.
    """
    try:
        if Path(RESULTS_FILE).exists():
            with open(RESULTS_FILE, 'r') as f:
                return json.load(f)
        return []
    except json.JSONDecodeError:
        logger.warning(f"Could not decode {RESULTS_FILE}. Starting with empty results.")
        return []
    except Exception as e:
        logger.error(f"Error loading results file: {str(e)}")
        return []

async def save_results(new_results):
    """
    Save new results to the JSON file by appending them to existing results.
    
    Args:
        new_results (dict): The new classification results to save
    """
    try:
        # Load existing results
        existing_results = await load_existing_results()
        
        # Add timestamp to new results
        new_results['timestamp'] = datetime.now().isoformat()
        
        # Append new results
        existing_results.append(new_results)
        
        # Write back to file with proper formatting
        with open(RESULTS_FILE, 'w') as f:
            json.dump(existing_results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Successfully saved results to {RESULTS_FILE}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Manages database connections and cleanup.
    """
    # Startup
    try:
        
        await mongodb.connect() 
        logger.info("Connected to MongoDB database")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
       
    
    yield
    
    # Shutdown
    # Close MongoDB connection
    await mongodb.disconnect()  # Add this method to your MongoDB class
    logger.info("Disconnected from MongoDB database")

# Create FastAPI app
app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)

# # Initialize email classifier
# classifier = ManualEmailClassifier(
#     prisma_client=prisma,
#     pinecone_index=pc.Index("email-embeddings"),
#     openai_client=openai_client
# )

# @app.post("/process-emails")
# async def process_emails(skip: int = 0, batch_size: int = 10):
#     """
#     Endpoint to process emails and return classification results immediately.
#     Results are both returned to the client and saved to a JSON file.
    
#     Args:
#         skip (int): Number of emails to skip (for pagination)
#         batch_size (int): Number of emails to process in this batch
#     """
#     try:
#         # Log the incoming parameters
#         logger.info(f"Processing emails with skip={skip}, batch_size={batch_size}")
        
#         # Process emails with the provided skip value
#         results = await classifier.classify_emails(batch_size=batch_size, skip=skip)
        
#         if not results or results.get("status") == "no_emails":
#             logger.info("No unprocessed emails found")
#             raise HTTPException(
#                 status_code=404,
#                 detail="No unprocessed emails found"
#             )
            
#         if results.get("status") == "error":
#             logger.error(f"Classification failed: {results.get('message', 'Unknown error')}")
#             raise HTTPException(
#                 status_code=500,
#                 detail=results.get("message", "Classification failed")
#             )
        
#         # Calculate the next skip value based on current skip + number of processed emails
#         processed_count = len(results.get("results", []))
#         next_skip = skip + processed_count
        
#         # Add the next skip value to the response
#         results["next_skip"] = next_skip
#         logger.info(f"Processed {processed_count} emails. Next skip value: {next_skip}")
        
#         # Save results to JSON file
#         await save_results(results)
            
#         return results
        
#     except Exception as e:
#         logger.error(f"Failed to process emails: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         )
@app.post("/process-emails-async")
async def process_emails_async(skip: int = 0, batch_size: int = 10):
    """
    Endpoint to start asynchronous email processing.
    Returns a task ID that can be used to check the status.
    """
    try:
        # Start celery task
        task = classify_email_batch.delay(batch_size, skip)
        logger.info(msg=f"this is the task id {task.id}")
        return {
            "status": "processing",
            "task_id": task.id,
            "message": "Email processing started"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start processing: {str(e)}"
        )

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
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

# Process multiple batches in parallel
@app.post("/process-email-batches")
async def process_email_batches(total_emails: int = 50, batch_size: int = 10, skip: int = 0):
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
            current_skip = original_skip + (i * batch_size)  # Use original skip as base
            task = classify_email_batch.delay(batch_size, current_skip)
            task_ids.append(task.id)
            
        return {
            "status": "processing",
            "task_ids": task_ids,
            "message": f"Started processing {num_batches} batches",
            "next_skip": original_skip + (num_batches * batch_size)  # Calculate next skip correctly
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start batch processing: {str(e)}"
        )
# api/endpoints.py
# api/endpoints.py
from fastapi import HTTPException
from typing import Optional
from mongodb import mongodb
import logging

logger = logging.getLogger(__name__)

@app.get("/classification-results")
async def get_classification_results(skip: int = 0, limit: int = 500):
    try:
        # Log the query parameters
        logger.info(f"Querying MongoDB with skip={skip}, limit={limit}")
        
        # Check collection existence and document count
        collection_names = await mongodb.db.list_collection_names()
        logger.info(f"Available collections: {collection_names}")
        
        total_docs = await mongodb.results.count_documents({})
        logger.info(f"Total documents in collection: {total_docs}")
        
        # Perform the query with detailed logging
        results = await mongodb.get_classification_results(skip=skip, limit=limit)
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

@app.get("/classification-results/category/{category}")
async def get_results_by_category(category: str, skip: int = 0, limit: int = 10):
    """
    Get classification results filtered by category
    """
    try:
        results = await mongodb.get_results_by_category(category, skip, limit)
        
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
@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)