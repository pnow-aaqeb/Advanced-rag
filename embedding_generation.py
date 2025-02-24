# from transformers import AutoTokenizer, AutoModel
# import torch
# from prisma import Prisma
# from pinecone import Pinecone
# import json
# import os
# from dotenv import load_dotenv
# import html2text
# import re
# import asyncio
# import httpx
# from celery_config import celery_app
# import time
# import random


# load_dotenv()
# html_converter = html2text.HTML2Text()
# # Configure the converter for optimal email content processing
# html_converter.ignore_links = False  # Keep links as they might be important
# html_converter.ignore_images = True  # Skip image data
# html_converter.ignore_tables = False  # Keep table content
# html_converter.body_width = 0  # Don't wrap text at specific width
# class TextEmbedder:
#     """
#     A class to generate and manage text embeddings using the BAAI/bge-large-en-v1.5 model.
#     This class handles the entire pipeline from text input to normalized embeddings.
#     """
#     def __init__(self, model_name='BAAI/bge-large-en-v1.5'):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         print(f"Using device: {self.device}")
#         self.model_name = model_name
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)
#         # self.html2text=html2text.HTML2Text()
#         # self.html2text.ignore_links=True
#         self.model.eval()
        
#     def get_embeddings_batch(self, texts, batch_size=32):
#         """
#         Process multiple texts in batches for better GPU utilization.
#         """
#         all_embeddings = []
        
#         # Process texts in batches
#         for i in range(0, len(texts), batch_size):
#             batch_texts = texts[i:i + batch_size]
#             prepared_texts = [f"Represent this sentence for retrieval: {text}" for text in batch_texts]
            
#             # Tokenize entire batch at once
#             inputs = self.tokenizer(
#                 prepared_texts,
#                 padding=True,
#                 truncation=True,
#                 max_length=512,
#                 return_tensors='pt'
#             )
            
#             # Move batch to GPU
#             # inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#                 # Process entire batch at once
#                 embeddings = torch.mean(outputs.last_hidden_state, dim=1)
#                 normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
#                 # Move results back to CPU
#                 normalized_embeddings = normalized_embeddings.cuda()
#                 all_embeddings.extend(normalized_embeddings)
        
#         return all_embeddings

#     def get_embedding(self, text):
#         """
#         Process a single text, but still using the GPU.
#         """
#         return self.get_embeddings_batch([text])[0]
#     def tensor_to_list(self, tensor):
#         """
#         Convert PyTorch tensor to a Python list for database storage.
        
#         Args:
#             tensor (torch.Tensor): The PyTorch tensor to convert
#         Returns:
#             list: The tensor converted to a Python list
#         """
#         return tensor.tolist()
# @celery_app.task(bind=True, max_retries=5, name='process_single_message')
# def process_single_message(self, message_data: dict):
#     """
#     A Celery task that processes a single message's embeddings.
#     Handles all components of an email: subject, body, sender, and recipients.
#     """
#     try:
#         embedder = TextEmbedder()
#         pc = Pinecone(api_key=os.getenv('PINCONE_API_KEY'))
#         index = pc.Index('email-embeddings')
        
#         embeddings = {}
        
#         # Generate embeddings (remaining code same as before)
#         if message_data['subject']:
#             embeddings['subject_embedding'] = embedder.tensor_to_list(
#                 embedder.get_embedding(message_data['subject'])
#             )
        
#         if message_data['body']:
#             processed_body = html_converter.handle(message_data['body'])
#             if processed_body:
#                 embeddings['body_embedding'] = embedder.tensor_to_list(
#                     embedder.get_embedding(processed_body)
#                 )
        
#         if message_data['sender_email']:
#             sender_text = f"{message_data['sender_name'] or ''} {message_data['sender_email']}".strip()
#             embeddings['sender_embedding'] = embedder.tensor_to_list(
#                 embedder.get_embedding(sender_text)
#             )
        
#         if message_data['recipients']:
#             recipient_list = (json.loads(message_data['recipients']) 
#                             if isinstance(message_data['recipients'], str) 
#                             else message_data['recipients'])
            
#             receiver_text = ", ".join([
#                 r.get('emailAddress', {}).get('address', '') 
#                 for r in recipient_list
#             ])
            
#             if receiver_text:
#                 embeddings['receiver_embedding'] = embedder.tensor_to_list(
#                     embedder.get_embedding(receiver_text)
#                 )
        
#         # Store in Pinecone with proper metadata handling
#         vectors = []
#         for embed_type, embedding in embeddings.items():
#             if embedding:
#                 # Prepare base metadata
#                 metadata = {
#                     "message_id": message_data['id'],
#                     "component": embed_type,
#                     "sender_email": message_data['sender_email'] or "",
#                     "subject": message_data['subject'] or "",
#                     "sent_date": message_data['sent_date_time'] or "",
#                 }
                
#                 # Only add recipients for receiver embeddings and ensure it's a string
#                 if embed_type == 'receiver_embedding' and message_data['recipients']:
#                     if isinstance(message_data['recipients'], str):
#                         metadata["recipients"] = message_data['recipients']
#                     else:
#                         metadata["recipients"] = json.dumps(message_data['recipients'])
                
#                 vectors.append({
#                     "id": f"{embed_type}_{message_data['id']}",
#                     "values": embedding,
#                     "metadata": metadata
#                 })
        
#         if vectors:
#             try:
#                 max_retries = 3
#                 retry_count = 0
#                 while retry_count < max_retries:
#                     try:
#                         # Split vectors into smaller batches
#                         batch_size = 100
#                         for i in range(0, len(vectors), batch_size):
#                             batch = vectors[i:i + batch_size]
#                             index.upsert(vectors=batch, namespace="email_embeddings")
#                             time.sleep(0.5)
#                         break
#                     except Exception as e:
#                         retry_count += 1
#                         if retry_count == max_retries:
#                             raise e
#                         wait_time = (2 ** retry_count) + random.uniform(0, 1)
#                         time.sleep(wait_time)
#             except Exception as e:
#                 print(f"Failed to upsert vectors after {max_retries} attempts: {str(e)}")
#                 raise
        
#         return {
#             'status': 'success', 
#             'message_id': message_data['id'],
#             'embeddings_generated': list(embeddings.keys())
#         }
        
#     except Exception as exc:
#         print(f"Error processing message {message_data['id']}: {str(exc)}")
#         retry_delay = 60 * (2 ** self.request.retries)
#         retry_delay += random.uniform(0, 30)  # Add jitter
#         self.retry(exc=exc, countdown=retry_delay, max_retries=5)
# class EmailEmbeddingGenerator:
#     """
#     A class to handle the generation and storage of email embeddings in the database.
#     This class uses TextEmbedder for generating embeddings and Prisma for database operations.
#     """
#     def __init__(self, batch_size=100,index_name='email-embeddings',start_from=1001):
#         self.embedder = TextEmbedder()
#         self.prisma = Prisma()
#         self.batch_size = batch_size
#         self.retry_delay = 5  # seconds between retries
#         self.pc = Pinecone(api_key=os.getenv('PINCONE_API_KEY'))
#         self.index = self.pc.Index(index_name)
#         self.start_from=start_from
        
#     async def connect(self):
#         """Establish database connection with retry logic."""
#         max_attempts = 3
#         for attempt in range(max_attempts):
#             try:
#                 await self.prisma.connect()
#                 print("Connected to database successfully")
#                 return
#             except Exception as e:
#                 if attempt < max_attempts - 1:
#                     wait_time = self.retry_delay * (attempt + 1)
#                     print(f"Connection attempt {attempt + 1} failed. Retrying in {wait_time}s...")
#                     await asyncio.sleep(wait_time)
#                 else:
#                     raise

    
#     def tensor_to_list(self, tensor):
#         """
#         Convert PyTorch tensor to a Python list for database storage.
#         PyTorch tensors cannot be directly stored in databases, so we need this conversion.
#         """
#         return tensor.tolist()
    
#     def process_recipients(self, recipients):
#         """
#         Process the recipients JSON data into a single string for embedding.
#         Handles both string JSON and parsed JSON objects.
#         """
#         if not recipients:
#             return ""
#         # Handle both string JSON and already parsed JSON
#         recipient_list = json.loads(recipients) if isinstance(recipients, str) else recipients
#         # Extract and combine email addresses
#         return ", ".join([r.get('emailAddress', {}).get('address', '') for r in recipient_list])
    
#     # async def generate_and_store_embeddings(self, message_id: str):
#     #     """
#     #     Generate embeddings and store them in both Prisma and Pinecone.
#     #     """
#     #     message = await self.prisma.messages.find_unique(
#     #         where={'id': message_id}
#     #     )
        
#     #     if not message:
#     #         raise ValueError(f"Message with ID {message_id} not found")
        
#     #     # Generate embeddings
#     #     embeddings = {
#     #         'subject_embedding': self.tensor_to_list(
#     #             self.embedder.get_embedding(message.subject or "")
#     #         ) if message.subject else None,
#     #         # ... (other embeddings remain the same)
#     #     }
        
#     #     # Store in both Prisma and Pinecone
#     #     # await self.prisma.messages.update(
#     #     #     where={'id': message_id},
#     #     #     data=embeddings
#     #     # )
        
#     #     # Prepare and store in Pinecone
#     #     vectors = self._prepare_for_pinecone(message_id, embeddings, message)
#     #     self.index.upsert(
#     #         vectors=vectors,
#     #         namespace="email_embeddings"
#     #     )
        
#     def _prepare_for_pinecone(self, message_id: str, embeddings: dict, message: dict):
#         """
#         Internal method to prepare embeddings for Pinecone storage.
#         Note the underscore prefix indicating this is an internal method.
#         """
#         vectors = []
        
#         def add_vector(embedding, prefix, metadata):
#             if embedding:
#                 vectors.append({
#                     "id": f"{prefix}_{message_id}",
#                     "values": embedding,
#                     "metadata": {
#                         "message_id": message_id,
#                         "component": prefix,
#                         "sender_email": message.sender_email,
#                         "subject": message.subject,
#                         "sent_date": message.sent_date_time.isoformat() if message.sent_date_time else None,
#                         **metadata
#                     }
#                 })
        
#         # Add vectors for each component
#         add_vector(embeddings.get('subject_embedding'), 'subject', {"type": "subject"})
#         add_vector(embeddings.get('body_embedding'), 'body', {"type": "body"})
#         add_vector(embeddings.get('sender_embedding'), 'sender', {"type": "sender"})
#         add_vector(embeddings.get('receiver_embedding'), 'receiver', {"type": "receiver"})
        
#         return vectors
#     def clean_text(self, text: str) -> str:
#         """
#         Clean and normalize text content to prepare it for embedding.
#         This method handles common email text artifacts and formatting issues.
        
#         Args:
#             text (str): Raw text content to clean
            
#         Returns:
#             str: Cleaned and normalized text
#         """
#         if not text:
#             return ""
            
#         # Convert multiple spaces to single space
#         text = re.sub(r'\s+', ' ', text)
        
#         # Remove email signature markers
#         text = re.sub(r'--+\s*$', '', text)
        
#         # Remove common email reply markers
#         text = re.sub(r'^>+\s*', '', text, flags=re.MULTILINE)
        
#         # Remove excessive newlines
#         text = re.sub(r'\n{3,}', '\n\n', text)
        
#         return text.strip()

#     def process_html_body(self, html_content: str) -> str:
#         """
#         Process HTML body content into clean plain text suitable for embedding.
#         This method handles the conversion from HTML to text and applies necessary cleanup.
        
#         Args:
#             html_content (str): Raw HTML content from email body
            
#         Returns:
#             str: Processed plain text ready for embedding
#         """
#         if not html_content:
#             return ""
            
#         try:
#             # Convert HTML to plain text
#             plain_text = html_converter.handle(html_content)
            
#             # Clean the resulting text
#             cleaned_text = self.clean_text(plain_text)
            
#             # Ensure we have reasonable length content
#             # We'll truncate very long content to the first 10000 characters
#             # This helps maintain embedding quality while managing processing time
#             return cleaned_text[:10000]
            
#         except Exception as e:
#             print(f"Error processing HTML content: {str(e)}")
#             # If HTML processing fails, try to extract text directly
#             # This serves as a fallback mechanism
#             return self.clean_text(html_content)[:10000]
        
#     async def process_batch(self, transaction, messages):
#         """
#         Processes a batch of messages by distributing them to Celery workers.
#         This method queues each message as a separate task for parallel processing.
#         """
#         processed_ids = []
#         failed_ids = []
#         tasks = []

#         # Queue each message as a separate task
#         for message in messages:
#             try:
#                 # Prepare the message data that will be sent to the Celery task
#                 message_data = {
#                     'id': message.id,
#                     'subject': message.subject,
#                     'body': message.body,
#                     'sender_email': message.sender_email,
#                     'sender_name': message.sender_name,
#                     'sent_date_time': message.sent_date_time.isoformat() if message.sent_date_time else None,
#                     'recipients': message.recipients
#                 }

#                 # Alternate between queues for load balancing
#                 queue_name = 'queue_1' if len(tasks) % 2 == 0 else 'queue_2'

#                 # Queue the task and store the task object
#                 task = process_single_message.apply_async(
#                     args=[message_data],
#                     queue=queue_name
#                 )
#                 tasks.append((message.id, task))
#                 print(f"Queued message {message.id} for processing")

#             except Exception as e:
#                 print(f"Error queueing message {message.id}: {str(e)}")
#                 failed_ids.append(message.id)

#         # Wait for all tasks in this batch to complete
#         for message_id, task in tasks:
#             try:
#                 result = task.get(timeout=3600)  # 1 hour timeout per task
#                 if result['status'] == 'success':
#                     processed_ids.append(message_id)
#                     print(f"Successfully processed message {message_id} with embeddings: {result['embeddings_generated']}")
#             except Exception as e:
#                 print(f"Task failed for message {message_id}: {str(e)}")
#                 failed_ids.append(message_id)

#         return processed_ids, failed_ids

#     async def process_all_messages(self):
#         """Process all messages in batches with transaction support."""
#         try:
#             async with self.prisma.tx(timeout=5000000) as transaction:
#                 # Get total count
#                 total_messages = await transaction.messages.count()
#                 print(f"Found {total_messages} messages to process")
                
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
#                             break
                            
#                         print(f"\nProcessing batch of {len(messages)} messages")
                        
#                         # Process batch
#                         processed_ids, failed_ids = await self.process_batch(transaction, messages)
                        
#                         # Update counts
#                         processed_count += len(processed_ids)
#                         failed_messages.extend(failed_ids)
                        
#                         # Show progress
#                         progress = (processed_count / total_messages) * 100
#                         print(f"Progress: {progress:.1f}% ({processed_count}/{total_messages})")
                        
#                     except httpx.ReadTimeout:
#                         print("Timeout occurred. Retrying batch...")
#                         await asyncio.sleep(self.retry_delay)
#                         continue
                        
#                 # Final summary
#                 print("\nProcessing completed:")
#                 print(f"Successfully processed: {processed_count}")
#                 print(f"Failed messages: {len(failed_messages)}")
#                 if failed_messages:
#                     print("Failed IDs:", failed_messages)

#         except Exception as e:
#             print(f"Fatal error during processing: {str(e)}")
#             raise

# # Example usage showing how to run the embedding generation
# if __name__ == "__main__":
#     async def main():
#         try:
#             # Initialize generator
#             generator = EmailEmbeddingGenerator(batch_size=100,start_from=1001)
            
#             # Connect to database
#             await generator.connect()
            
#             # Process messages
#             await generator.process_all_messages()
            
#         except Exception as e:
#             print(f"Error in main process: {str(e)}")
#         finally:
#             # Ensure we disconnect from the database
#             await generator.prisma.disconnect()

#     # Run the async main function
#     import asyncio
#     asyncio.run(main())