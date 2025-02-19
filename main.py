# from contextlib import asynccontextmanager
# from fastapi import FastAPI, BackgroundTasks
# from prisma import Prisma
# from pinecone import Pinecone
# import logging
# from email_processor import EmailProcessor 
# from dotenv import load_dotenv
# import os
# from fastapi.middleware.cors import CORSMiddleware

# # Load environment variables and configure logging
# load_dotenv()
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize connections
# prisma = Prisma()
# pc = Pinecone(api_key=os.getenv('PINCONE_API_KEY'))

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Handles application startup and shutdown events.
#     Manages database connections and cleanup.
#     """
#     # Startup
#     await prisma.connect()
#     logger.info("Connected to Prisma database")
    
#     yield
    
#     # Shutdown
#     await prisma.disconnect()
#     logger.info("Disconnected from Prisma database")

# # Create FastAPI app
# app = FastAPI(lifespan=lifespan)

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize email processor
# processor = EmailProcessor(
#     pinecone_client=pc,
#     prisma_client=prisma,
#     openai_api_key=os.getenv('OPENAI_API_KEY'),
#     business_context="ProficientNow is a recruitment company"
# )

# @app.post("/process-emails")
# async def process_emails(background_tasks: BackgroundTasks):
#     """
#     Endpoint to trigger email processing.
#     Uses background tasks to handle processing asynchronously.
#     """
#     if processor.is_processing:
#         return {"status": "already_running"}
        
#     background_tasks.add_task(processor.process_all_messages)
#     return {"status": "started"}

# # Add this if you want to run directly with Python
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
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
    allow_origins=["http://localhost"],
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
async def process_emails(background_tasks: BackgroundTasks):
    """
    Endpoint to trigger email processing.
    Uses background tasks to handle processing asynchronously.
    """
    try:
        background_tasks.add_task(classifier.classify_emails)
        return {"status": "processing_started"}
    except Exception as e:
        logger.error(f"Failed to start email processing: {str(e)}")
        return {"status": "error", "message": str(e)}

# @app.get("/status")
# async def get_status():
#     """
#     Endpoint to check processing status
#     """
#     return {"is_processing": classifier.is_processing}

# Add this if you want to run directly with Python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)