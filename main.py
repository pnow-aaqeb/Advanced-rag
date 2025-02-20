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

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize connections
prisma = Prisma()
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    Endpoint to process emails and return classification results immediately
    """
    try:
        # Process next batch of emails and get results
        results = await classifier.classify_emails(batch_size=1)
        
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