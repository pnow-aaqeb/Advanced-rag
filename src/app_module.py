from contextlib import asynccontextmanager
from nest.core import PyNestFactory, Module
from .app_controller import AppController
from .app_service import AppService
from .email_classification.email_classification_module import EmailClassificationModule
from src.database.mongodb import MongoDB
import logging
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    print("Starting application...")
    
    # Create a MongoDB instance and connect it
    # This will be the singleton instance used throughout the application
    mongodb = MongoDB()
    await mongodb.connect()
    print("MongoDB connection established")
    
    # Store it in app state so it can be accessed if needed
    app.state.mongodb = mongodb
    
    yield
    
    print("Shutting down application...")
    if hasattr(app.state, 'mongodb'):
        await app.state.mongodb.disconnect()
        print("MongoDB connection closed")

@Module(
    imports=[EmailClassificationModule], 
    controllers=[AppController], 
    providers=[AppService, MongoDB]  # Add MongoDB here!
)
class AppModule:
    pass

app = PyNestFactory.create(
    AppModule,
    description="This is my PyNest app.",
    title="PyNest Application",
    version="1.0.0",
    debug=True,
)

http_server = app.get_server()
http_server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

http_server.router.lifespan_context = lifespan