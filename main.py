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

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize connections
prisma = Prisma()
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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
    await prisma.connect()
    logger.info("Connected to Prisma database")
    
    # Initialize Pinecone index
    index = pc.Index("email-embeddings")
    logger.info("Connected to Pinecone index")
    
    yield
    
    # Shutdown
    await prisma.disconnect()
    logger.info("Disconnected from Prisma database")

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

# Initialize email classifier
classifier = ManualEmailClassifier(
    prisma_client=prisma,
    pinecone_index=pc.Index("email-embeddings"),
    openai_client=openai_client
)

@app.post("/process-emails")
async def process_emails():
    """
    Endpoint to process emails and return classification results immediately.
    Results are both returned to the client and saved to a JSON file.
    """
    try:
        # Process next batch of emails and get results
        results = await classifier.classify_emails(batch_size=10, skip=110)
        
        if not results or results.get("status") == "no_emails":
            raise HTTPException(
                status_code=404,
                detail="No unprocessed emails found"
            )
            
        if results.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=results.get("message", "Classification failed")
            )
        
        # Save results to JSON file
        await save_results(results)
            
        return results
        
    except Exception as e:
        logger.error(f"Failed to process emails: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
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