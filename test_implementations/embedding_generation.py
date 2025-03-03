# import torch
# import numpy as np
# import os
# import logging
# import re
# import asyncio
# import httpx
# import time
# import random
# import json
# from typing import Tuple, Dict, List, Any, Optional
# from dotenv import load_dotenv
# import html2text
# from prisma import Prisma
# from pinecone import Pinecone
# from celery_config import celery_app

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# # Initialize HTML converter once
# html_converter = html2text.HTML2Text()
# html_converter.ignore_images = True
# html_converter.ignore_links = False
# html_converter.ignore_tables = False
# html_converter.body_width = 0

# # Get worker type from environment - either "gpu" or "cpu"
# WORKER_TYPE = os.getenv("WORKER_TYPE", "cpu").lower()
# logger.info(f"Worker type set to: {WORKER_TYPE}")

# #######################
# # Embedding Service
# #######################

# class EmbeddingService:
#     """A service that provides embeddings using either a local model or a remote API."""
    
#     @staticmethod
#     def get_embedding(text: str) -> List[float]:
#         """
#         Generate an embedding for the given text using the appropriate method based on worker type.
#         This is a facade that delegates to either GPU, CPU, or API-based embedding.
        
#         Args:
#             text: The text to embed
            
#         Returns:
#             List[float]: The embedding vector
#         """
#         # Handle empty input
#         if not text or not text.strip():
#             # Return a zero vector of the expected dimension 
#             return [0.0] * 1024  # BGE-large-en-v1.5 has 1024 dimensions
            
#         try:
#             if WORKER_TYPE == "gpu":
#                 # GPU worker - use the model directly
#                 return GPUEmbeddingProvider.get_embedding(text)
#             else:
#                 # CPU worker - use CPU-based model
#                 return CPUEmbeddingProvider.get_embedding(text)
#         except Exception as e:
#             logger.error(f"Error generating embedding: {str(e)}")
#             # Fallback to CPU in case of GPU errors
#             if WORKER_TYPE == "gpu":
#                 logger.warning("GPU embedding failed, falling back to CPU")
#                 try:
#                     return CPUEmbeddingProvider.get_embedding(text)
#                 except Exception as cpu_err:
#                     logger.error(f"CPU fallback also failed: {str(cpu_err)}")
#                     raise
#             else:
#                 raise

# class GPUEmbeddingProvider:
#     """Provider that uses GPU for embeddings - only used by GPU workers."""
    
#     _model = None
#     _tokenizer = None
    
#     @classmethod
#     def _load_model(cls):
#         """Load the BGE model with CUDA support."""
#         if cls._model is None:
#             try:
#                 from transformers import AutoTokenizer, AutoModel
                
#                 # Set PyTorch to use less memory
#                 os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
                
#                 logger.info("Loading BGE model on GPU...")
#                 model_name = 'BAAI/bge-large-en-v1.5'
#                 cls._tokenizer = AutoTokenizer.from_pretrained(model_name)
#                 cls._model = AutoModel.from_pretrained(model_name)
                
#                 # Move to GPU with half precision to save memory
#                 cls._model = cls._model.to("cuda").half()  # Use half precision
#                 cls._model.eval()  # Set to evaluation mode
#                 logger.info("BGE model loaded successfully on GPU")
#             except Exception as e:
#                 logger.error(f"Failed to load BGE model on GPU: {str(e)}")
#                 raise
    
#     @classmethod
#     def get_embedding(cls, text: str) -> List[float]:
#         """Generate embedding using GPU."""
#         cls._load_model()
        
#         # Prepare input for BGE model format
#         prepared_text = f"Represent this sentence for retrieval: {text}"
        
#         # Use GPU for inference with reduced precision
#         with torch.cuda.amp.autocast():  # Use automatic mixed precision
#             inputs = cls._tokenizer(prepared_text, padding=True, truncation=True, 
#                                   max_length=512, return_tensors='pt').to("cuda")
            
#             with torch.no_grad():
#                 outputs = cls._model(**inputs)
#                 embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                
#         # Normalize embeddings
#         norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
#         normalized_embeddings = embeddings / norm
        
#         # Clear GPU cache after processing
#         torch.cuda.empty_cache()
        
#         return normalized_embeddings[0].tolist()

# class CPUEmbeddingProvider:
#     """Provider that uses CPU for embeddings - used by most workers."""
    
#     _model = None
#     _tokenizer = None
    
#     @classmethod
#     def _load_model(cls):
#         """Load the BGE model on CPU."""
#         if cls._model is None:
#             try:
#                 from transformers import AutoTokenizer, AutoModel
                
#                 logger.info("Loading BGE model on CPU...")
#                 model_name = 'BAAI/bge-large-en-v1.5'
#                 cls._tokenizer = AutoTokenizer.from_pretrained(model_name)
#                 cls._model = AutoModel.from_pretrained(model_name)
                
#                 # Ensure we're on CPU
#                 cls._model = cls._model.to("cpu")
#                 cls._model.eval()  # Set to evaluation mode
#                 logger.info("BGE model loaded successfully on CPU")
#             except Exception as e:
#                 logger.error(f"Failed to load BGE model on CPU: {str(e)}")
#                 raise
    
#     @classmethod
#     def get_embedding(cls, text: str) -> List[float]:
#         """Generate embedding using CPU."""
#         cls._load_model()
        
#         # Prepare input for BGE model format
#         prepared_text = f"Represent this sentence for retrieval: {text}"
#         inputs = cls._tokenizer(prepared_text, padding=True, truncation=True, 
#                               max_length=512, return_tensors='pt')
        
#         with torch.no_grad():
#             outputs = cls._model(**inputs)
#             embeddings = outputs.last_hidden_state[:, 0].numpy()
            
#         # Normalize embeddings
#         norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
#         normalized_embeddings = embeddings / norm
        
#         return normalized_embeddings[0].tolist()

# #######################
# # Pinecone Singleton
# #######################

# class PineconeSingleton:
#     _instance = None
#     _initialized = False

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance

#     def __init__(self):
#         if not self._initialized:
#             try:
#                 api_key = os.getenv("PINECONE_API_KEY")
#                 if not api_key:
#                     raise ValueError("PINECONE_API_KEY environment variable not set")
                    
#                 self.pc = Pinecone(api_key=api_key)
#                 self.index = self.pc.Index("embeddings2")
#                 logger.info("Successfully initialized Pinecone connection")
#                 self._initialized = True
#             except Exception as e:
#                 logger.error(f"Failed to initialize Pinecone: {str(e)}")
#                 raise

#     def get_index(self):
#         if not self._initialized:
#             raise RuntimeError("Pinecone singleton not properly initialized")
#         return self.index

# #######################
# # Email Processing
# #######################

# class EmailContentProcessor:
#     """
#     Improved email parser that precisely extracts the most recent email body from a thread.
#     Eliminates noise from quoted content and accurately isolates just the current message.
#     """
    
#     @staticmethod
#     def extract_current_email(email_content: str, is_html: bool = True) -> Tuple[str, dict]:
#         """
#         Extracts only the current/most recent email body from a thread.
        
#         Args:
#             email_content: The raw email content (HTML or text)
#             is_html: Whether the content is in HTML format
            
#         Returns:
#             Tuple containing:
#                 - The extracted current email body text
#                 - Metadata dictionary with email details (sender, time, etc.)
#         """
#         # Handle empty content
#         if not email_content:
#             return "", {}
            
#         # Convert HTML to text if needed
#         if is_html:
#             try:
#                 text_content = html_converter.handle(email_content)
#             except Exception as e:
#                 logger.warning(f"HTML conversion failed: {str(e)}. Treating as plain text.")
#                 text_content = email_content
#         else:
#             text_content = email_content
            
#         # Common delimiter patterns that indicate the start of quoted/previous emails
#         delimiter_patterns = [
#             # Most common email client patterns
#             r'On\s+.*?,\s+.*?wrote:',                       # Gmail/Apple style - On [date], [name] wrote:
#             r'On\s+.*?,\s+.*?<.*?>\s+wrote:',               # Variation with email
#             r'On\s+.*?\s+at\s+.*?,\s+.*?wrote:',            # With time - On Dec 25 at 10:00 AM, John wrote:
#             r'From:[\s\S]*?Sent:[\s\S]*?To:[\s\S]*?Subject:', # Outlook header format
#             r'From:\s*.*?\s*\[mailto:.*?\]',                # Outlook-style with mailto
#             r'-+\s*Original Message\s*-+',                  # Original message headers
#             r'Begin forwarded message:',                     # Forwarded message marker
#             r'Forwarded message\s*-+',                       # Another forwarded style
#             r'Reply all\s*Reply\s*Forward',                  # Email client action buttons
#             r'_+',                                          # Horizontal line separator (often used as delimiter)
#             r'>{3,}',                                        # Multiple quote markers
#             r'From:.*?Subject:.*?\n\n',                     # Generic email header pattern
#             r'Sent from my iPhone',                          # Mobile signatures (treat as end of message)
#             r'Sent from my Android',
#             r'\*From:\*',                                   # Some HTML to text conversions of bold headers
#         ]
        
#         # Create a single regex pattern from all delimiter patterns
#         combined_pattern = '|'.join(f'({pattern})' for pattern in delimiter_patterns)
        
#         # Find all matches
#         matches = list(re.finditer(combined_pattern, text_content, re.MULTILINE | re.DOTALL))
        
#         # Extract metadata
#         metadata = {}
#         sender_match = re.search(r'From:\s*(.*?)(?=\s*<?[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}>?)', text_content)
#         if sender_match:
#             metadata['sender'] = sender_match.group(1).strip()
            
#         email_match = re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text_content)
#         if email_match:
#             metadata['email'] = email_match.group(0)
            
#         subject_match = re.search(r'Subject:\s*(.*?)(?=\n)', text_content)
#         if subject_match:
#             metadata['subject'] = subject_match.group(1).strip()
        
#         # If no delimiter is found, this is likely a single email (not a thread)
#         if not matches:
#             # Clean up the text
#             clean_text = re.sub(r'\n{3,}', '\n\n', text_content)  # Normalize multiple newlines
#             clean_text = re.sub(r'^\s+', '', clean_text, flags=re.MULTILINE)  # Remove leading whitespace
#             clean_text = EmailContentProcessor._process_markdown_for_display(clean_text)
#             return clean_text.strip(), metadata
            
#         # If delimiters are found, extract the content before the first one
#         first_match = min(matches, key=lambda m: m.start())
#         current_email = text_content[:first_match.start()].strip()
        
#         # Post-processing to clean the extracted content
#         # Remove header lines that might appear at the beginning of the current email
#         current_email = re.sub(r'^(?:From|To|Cc|Subject|Date):.*\n', '', current_email, flags=re.MULTILINE)
        
#         # Remove trailing signature markers
#         signature_patterns = [
#             r'--\s*$',                   # -- (common signature separator)
#             r'Best regards,[\s\S]*$',    # Common closing
#             r'Regards,[\s\S]*$',         # Another common closing
#             r'Thank you,[\s\S]*$',       # Thank you closing
#             r'Thanks,[\s\S]*$',          # Thanks closing
#             r'Sent from my (?:iPhone|iPad|Android|Galaxy|Samsung|mobile device)[\s\S]*$'  # Mobile signatures
#         ]
        
#         for pattern in signature_patterns:
#             current_email = re.sub(pattern, '', current_email, flags=re.DOTALL)
        
#         # Clean up any remaining whitespace issues
#         current_email = re.sub(r'\n{3,}', '\n\n', current_email)  # Normalize multiple newlines
#         current_email = current_email.strip()
        
#         # Handle markdown formatting
#         current_email = EmailContentProcessor._process_markdown_for_display(current_email)
        
#         return current_email, metadata
    
#     @staticmethod
#     def _process_markdown_for_display(text: str) -> str:
#         """
#         Process markdown formatting for display purposes.
#         This either renders markdown or escapes it based on the implementation needs.
#         """
#         if not text:
#             return ""
            
#         # Handle bold text (** or __)
#         text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
#         text = re.sub(r'__(.*?)__', r'\1', text)
        
#         # Handle italic text (* or _)
#         text = re.sub(r'\*(.*?)\*', r'\1', text)
#         text = re.sub(r'_(.*?)_', r'\1', text)
        
#         # Handle bullet points
#         text = re.sub(r'^\s*\*\s', '• ', text, flags=re.MULTILINE)  # Replace * with bullet
#         text = re.sub(r'^\s*-\s', '• ', text, flags=re.MULTILINE)   # Replace - with bullet
        
#         # Handle numbered lists
#         # Keep the numbers but standardize format
#         text = re.sub(r'^\s*(\d+)\.\s', r'\1. ', text, flags=re.MULTILINE)
        
#         # Handle links - extract just the link text
#         text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        
#         return text
        
#     @staticmethod
#     def is_empty_or_minimal_content(email_body: str) -> bool:
#         """
#         Checks if an email has empty or minimal content (like just a signature).
#         """
#         if not email_body or email_body.isspace():
#             return True
            
#         # Remove common signatures and links
#         stripped_body = email_body
#         signature_patterns = [
#             r'Get \[Outlook for iOS\].*',
#             r'Sent from my (?:iPhone|iPad|Android|Galaxy|Samsung|mobile device)',
#             r'--\s*.*',
#             r'Regards,.*$',
#             r'Thank you,.*$',
#             r'Best,.*$',
#         ]
        
#         for pattern in signature_patterns:
#             stripped_body = re.sub(pattern, '', stripped_body, flags=re.DOTALL)
            
#         # Count meaningful words (longer than 2 characters)
#         meaningful_words = [word for word in re.findall(r'\b\w+\b', stripped_body) if len(word) > 2]
        
#         # If we have fewer than 5 meaningful words, it's minimal content
#         return len(meaningful_words) < 5

# #######################
# # Celery Task
# #######################

# @celery_app.task(bind=True, max_retries=5, name='process_single_message')
# def process_single_message(self, message_data: dict):
#     """
#     A Celery task that processes a single message's embeddings.
#     Uses CPU or GPU based on worker type.
#     """
#     start_time = time.time()
#     try:
#         # Use Pinecone singleton
#         pinecone_singleton = PineconeSingleton()
#         index = pinecone_singleton.get_index()
        
#         # Process the email body to extract just the current email
#         email_processor = EmailContentProcessor()
#         clean_body, _ = email_processor.extract_current_email(message_data['body']) if message_data['body'] else ("", {})
        
#         # Check if the email is just minimal content
#         is_minimal = EmailContentProcessor.is_empty_or_minimal_content(clean_body)
        
#         # Skip or mark emails with minimal content
#         if is_minimal:
#             logger.info(f"Message {message_data['id']} contains minimal content, marking accordingly")
#             # Continue processing but add a metadata flag
#             minimal_content_flag = True
#         else:
#             minimal_content_flag = False
        
#         embeddings = {}
        
#         # Generate embeddings using the appropriate service
#         if message_data['subject']:
#             embeddings['subject_embedding'] = EmbeddingService.get_embedding(message_data['subject'])
        
#         if clean_body:
#             embeddings['body_embedding'] = EmbeddingService.get_embedding(clean_body)
        
#         if message_data['sender_email']:
#             sender_text = f"{message_data['sender_name'] or ''} {message_data['sender_email']}".strip()
#             embeddings['sender_embedding'] = EmbeddingService.get_embedding(sender_text)
        
#         # Process recipients
#         recipient_emails = []
#         if message_data['recipients']:
#             try:
#                 # Parse recipients to extract email addresses
#                 recipient_list = (json.loads(message_data['recipients']) 
#                                 if isinstance(message_data['recipients'], str) 
#                                 else message_data['recipients'])
                
#                 recipient_emails = [r.get('emailAddress', {}).get('address', '') 
#                                   for r in recipient_list 
#                                   if 'emailAddress' in r and 'address' in r.get('emailAddress', {})]
                
#                 receiver_text = ", ".join(recipient_emails)
                
#                 if receiver_text:
#                     embeddings['receiver_embedding'] = EmbeddingService.get_embedding(receiver_text)
#             except Exception as e:
#                 logger.warning(f"Error processing recipients for message {message_data['id']}: {str(e)}")
        
#         # Store in Pinecone with enhanced metadata
#         vectors = []
#         for embed_type, embedding in embeddings.items():
#             if embedding:
#                 # Prepare base metadata
#                 vector_metadata = {
#                     "message_id": message_data['id'],
#                     "component": embed_type,
#                     "sender_email": message_data['sender_email'] or "",
#                     "subject": message_data['subject'] or "",
#                     "sent_date": message_data['sent_date_time'] or "",
#                     "minimal_content": minimal_content_flag,
#                 }
                
#                 # Add truncated clean body to metadata (limited to avoid size issues)
#                 if clean_body:
#                     # Store a snippet of the body text (first 500 characters)
#                     vector_metadata["body_snippet"] = clean_body[:500] if len(clean_body) > 500 else clean_body
                
#                 # Add recipients information
#                 if recipient_emails:
#                     # Store recipient emails in metadata - ensure the list is properly serialized
#                     vector_metadata["recipient_emails"] = recipient_emails[:5]  # Limit to first 5 to avoid metadata size issues
                
#                 vectors.append({
#                     "id": f"{embed_type}_{message_data['id']}",
#                     "values": embedding,
#                     "metadata": vector_metadata
#                 })
        
#         if vectors:
#             try:
#                 max_retries = 3
#                 retry_count = 0
#                 while retry_count < max_retries:
#                     try:
#                         # Split vectors into smaller batches
#                         batch_size = 50  # Reduced batch size for reliability
#                         for i in range(0, len(vectors), batch_size):
#                             batch = vectors[i:i + batch_size]
#                             index.upsert(vectors=batch, namespace="embeddings2")
#                             time.sleep(0.1)  # Reduced sleep time
#                         break
#                     except Exception as e:
#                         retry_count += 1
#                         if retry_count == max_retries:
#                             raise e
#                         wait_time = (2 ** retry_count) + random.uniform(0, 1)
#                         time.sleep(wait_time)
#             except Exception as e:
#                 logger.error(f"Failed to upsert vectors after {max_retries} attempts: {str(e)}")
#                 raise
        
#         processing_time = time.time() - start_time
#         logger.info(f"Processed message {message_data['id']} in {processing_time:.2f}s using {WORKER_TYPE} worker")
        
#         return {
#             'status': 'success', 
#             'message_id': message_data['id'],
#             'embeddings_generated': list(embeddings.keys()),
#             'minimal_content': minimal_content_flag,
#             'processing_time': processing_time,
#             'worker_type': WORKER_TYPE
#         }
        
#     except Exception as exc:
#         logger.error(f"Error processing message {message_data['id']}: {str(exc)}")
#         retry_delay = 60 * (2 ** self.request.retries)
#         retry_delay += random.uniform(0, 30)  # Add jitter
#         self.retry(exc=exc, countdown=retry_delay, max_retries=5)

# #######################
# # Main Generator Class
# #######################

# class EmailEmbeddingGenerator:
#     """
#     A class to handle the generation and storage of email embeddings.
#     Uses singleton patterns for optimal resource usage.
#     """
#     def __init__(self, batch_size=100, start_from=0, max_gpu_batch=10):
#         self.prisma = Prisma()
#         self.batch_size = batch_size
#         self.max_gpu_batch = max_gpu_batch  # Control GPU memory usage
#         self.retry_delay = 5  # seconds between retries
#         self.start_from = start_from
#         self.email_processor = EmailContentProcessor()
        
#     async def connect(self):
#         """Establish database connection with retry logic."""
#         max_attempts = 3
#         for attempt in range(max_attempts):
#             try:
#                 await self.prisma.connect()
#                 logger.info("Connected to database successfully")
#                 return
#             except Exception as e:
#                 if attempt < max_attempts - 1:
#                     wait_time = self.retry_delay * (attempt + 1)
#                     logger.warning(f"Connection attempt {attempt + 1} failed. Retrying in {wait_time}s...")
#                     await asyncio.sleep(wait_time)
#                 else:
#                     raise
    
#     async def process_batch(self, transaction, messages):
#         """
#         Processes a batch of messages by distributing them to Celery workers.
#         Uses routing to ensure only GPU tasks go to GPU workers.
#         """
#         processed_ids = []
#         failed_ids = []
#         tasks = []

#         # Queue processing tasks with rate limiting and proper routing
#         for i, message in enumerate(messages):
#             try:
#                 # Prepare the message data for the Celery task
#                 message_data = {
#                     'id': message.id,
#                     'subject': message.subject,
#                     'body': message.body,
#                     'sender_email': message.sender_email,
#                     'sender_name': message.sender_name,
#                     'sent_date_time': message.sent_date_time.isoformat() if hasattr(message, 'sent_date_time') and message.sent_date_time else None,
#                     'recipients': message.recipients
#                 }

#                 # Determine which queue to use
#                 # This allows us to route tasks properly between GPU and CPU workers
#                 if i % 10 == 0:  # Every 10th message goes to GPU for better quality
#                     queue_name = "gpu_queue"
#                 else:
#                     # Distribute CPU tasks across multiple queues
#                     cpu_queue_index = (i % 9) + 1  # Use queues 1-9 for CPU
#                     queue_name = f"cpu_queue_{cpu_queue_index}"

#                 # Queue the task with routing to the right worker type
#                 task = process_single_message.apply_async(
#                     args=[message_data],
#                     queue=queue_name
#                 )
#                 tasks.append((message.id, task))
#                 logger.info(f"Queued message {message.id} for processing in {queue_name}")
                
#                 # Rate limiting to prevent CPU spikes
#                 if (i + 1) % self.max_gpu_batch == 0:
#                     await asyncio.sleep(0.5)  # Short pause every max_gpu_batch tasks
#                 else:
#                     await asyncio.sleep(0.05)  # Brief pause between tasks

#             except Exception as e:
#                 logger.error(f"Error queueing message {message.id}: {str(e)}", exc_info=True)
#                 failed_ids.append(message.id)

#         # Wait for tasks to complete
#         logger.info(f"Waiting for {len(tasks)} queued tasks to complete")
#         for message_id, task in tasks:
#             try:
#                 result = task.get(timeout=1800)  # 30 minute timeout per task
#                 if result['status'] == 'success':
#                     processed_ids.append(message_id)
#                     is_minimal = result.get('minimal_content', False)
#                     embeddings = result.get('embeddings_generated', [])
#                     worker_type = result.get('worker_type', 'unknown')
#                     processing_time = result.get('processing_time', 0)
#                     logger.info(f"Processed message {message_id} - Worker: {worker_type}, Time: {processing_time:.2f}s, Embeddings: {embeddings}, Minimal: {is_minimal}")
#             except Exception as e:
#                 logger.error(f"Task failed for message {message_id}: {str(e)}")
#                 failed_ids.append(message_id)

#         return processed_ids, failed_ids

#     async def process_all_messages(self):
#         """Process all messages in batches with improved resource management."""
#         try:
#             # Initialize Pinecone singleton in the main process
#             _ = PineconeSingleton()
            
#             async with self.prisma.tx(timeout=5000000) as transaction:
#                 # Get total count
#                 total_messages = await transaction.messages.count()
#                 logger.info(f"Found {total_messages} messages to process")
                
#                 processed_count = self.start_from
#                 failed_messages = []
                
#                 # Process in batches
#                 while processed_count < total_messages:
#                     try:
#                         # Fetch next batch
#                         messages = await transaction.messages.find_many(
#                             skip=processed_count,
#                             take=self.batch_size,
#                             order={'sent_date_time': 'desc'}
#                         )
                        
#                         if not messages:
#                             logger.info("No more messages to process")
#                             break
                            
#                         logger.info(f"Processing batch of {len(messages)} messages")
                        
#                         # Process batch
#                         processed_ids, failed_ids = await self.process_batch(transaction, messages)
                        
#                         # Update counts
#                         processed_count += len(processed_ids)
#                         failed_messages.extend(failed_ids)
                        
#                         # Show progress
#                         progress = (processed_count / total_messages) * 100
#                         logger.info(f"Progress: {progress:.1f}% ({processed_count}/{total_messages})")
                        
#                         # Brief pause between batches to let system resources recover
#                         await asyncio.sleep(1.0)
                        
#                     except httpx.ReadTimeout:
#                         logger.warning("Timeout occurred. Retrying batch...")
#                         await asyncio.sleep(self.retry_delay)
#                         continue
                        
#                 # Final summary
#                 logger.info("\nProcessing completed:")
#                 logger.info(f"Successfully processed: {processed_count}")
#                 logger.info(f"Failed messages: {len(failed_messages)}")
#                 if failed_messages:
#                     logger.info("Failed IDs: %s", str(failed_messages))

#         except Exception as e:
#             logger.error(f"Fatal error during processing: {str(e)}", exc_info=True)
#             raise

# # Example usage
# if __name__ == "__main__":
#     async def main():
#         try:
#             # Initialize generator with smaller batch size to manage resources better
#             generator = EmailEmbeddingGenerator(batch_size=100, start_from=0, max_gpu_batch=10)
            
#             # Connect to database
#             await generator.connect()
            
#             # Process messages
#             await generator.process_all_messages()
            
#         except Exception as e:
#             logger.error(f"Error in main process: {str(e)}", exc_info=True)
#         finally:
#             # Ensure we disconnect from the database
#             await generator.prisma.disconnect()

#     # Run the async main function
#     import asyncio
#     asyncio.run(main())