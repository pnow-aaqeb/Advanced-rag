import os
from pinecone import Pinecone
import logging

logger = logging.getLogger(__name__)

class PineconeSingleton:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            try:
                self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                self.index = self.pc.Index("email-embeddings")
                logger.info("Successfully initialized Pinecone connection")
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone: {str(e)}")
                raise

    def get_index(self):
        if not self._initialized:
            raise RuntimeError("Pinecone singleton not properly initialized")
        return self.index