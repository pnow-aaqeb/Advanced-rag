import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
import logging

logger = logging.getLogger(__name__)

class BGESingleton:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            try:
                self.model_name = 'BAAI/bge-large-en-v1.5'
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = self.model.to(self.device)
                logger.info(f"Successfully initialized BGE model on device: {self.device}")
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize BGE model: {str(e)}")
                raise

    def get_embedding(self, text: str) -> list:
        try:
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norm
            return normalized_embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

class BGEEmbeddings(Embeddings):
    def __init__(self):
        try:
            self.bge = BGESingleton()
        except Exception as e:
            logger.error(f"Failed to initialize BGEEmbeddings: {str(e)}")
            raise
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return [self.bge.get_embedding(text) for text in texts]
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.bge.get_embedding(text)
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise