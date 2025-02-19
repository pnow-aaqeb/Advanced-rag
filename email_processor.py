import asyncio
import logging
from langchain.chains.flare.base import FlareChain
from langchain_community.vectorstores import Pinecone
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
import torch
from typing import List
import numpy as np
import os
# First, let's keep our BGE embeddings class since it's essential for the vector store

business_context=''' ProficientNow Sales Pipeline Stages
1. Prospect Stage
Description:
Research team finds possible client companies
Basic information gathering about the company
Visibility:
Fully Visible (through internal research documents and emails)
2. Lead Generation Stage
Description:
Send first emails to companies
Follow up if they don't reply
If company replies, talk about services
Basic information exchange
Visibility:
Fully Visible (through email chains and internal updates)
3. Opportunity Stage
Description:
Transfer from Research team to Business Development Manager (BDM)
Detailed client requirement gathering by the BDM
Visibility:
Fully Visible (emails visible)
4. Fulfillment Stage
Description:
Assignment of recruiters
Active candidate sourcing
Candidate screening and shortlisting
Profile sharing with the client
Interview coordination
Candidate preparation
Interview feedback collection
Visibility:
Mix of Fully Visible (emails) and Not Visible (candidate calls and text messages)
5. Deal Stage
Description:
Contract negotiation
Terms and conditions discussion
Payment terms finalization
Service level agreements
Legal documentation
Contract signing
Visibility:
Fully Visible (through contract emails and documentation)
6. Sale Stage
Description:
Successful candidate placement
Offer letter issuance
Candidate joining confirmation
Invoice generation
Payment collection
Receipt acknowledgment
Post-placement follow-up
Visibility:
Fully Visible (through emails and system documentation) ''' 
class BGEEmbeddings(Embeddings):
    def __init__(self, model_name: str = 'BAAI/bge-large-en-v1.5'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        
    def _get_embedding(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norm
        return normalized_embeddings[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)

# Now, let's create a simplified EmailProcessor
class EmailProcessor:
    def __init__(self, pinecone_client, prisma_client, openai_api_key, business_context, model_name='BAAI/bge-large-en-v1.5'):
        # Initialize basic components
        self.prisma = prisma_client
        self.is_processing = False
        self.business_context = business_context
        
        
        # Set up embeddings and language models
        self.embeddings = BGEEmbeddings(model_name=model_name)
        self.lightweight_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=200,api_key=os.getenv('OPENAI_API_KEY'))
        self.llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=os.getenv('OPENAI_API_KEY'))
        
        # Set up vector store and retriever
        self.vectorstore = Pinecone.from_existing_index(
            index_name="email-embeddings",
            embedding=self.embeddings,
            text_key="text"
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': 10})
        
        # Initialize FLARE chain
        self.setup_flare_chain()
    def _verify_pinecone(self):
        print("Pinecone info:", self.vectorstore._index.describe_index_stats())

    def setup_flare_chain(self):
        # Use proper input variable names expected by FlareChain
        question_prompt = ChatPromptTemplate.from_messages([
            ("system", f"Identify uncertainties in emails. "),
            ("human", "Email content:\n{input}")
        ])
        
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", f"Classify the emails under the following sales stages: {business_context}"),
            ("human", "Email content:\n{input}")
        ])
        
        self.flare_chain = FlareChain.from_llm(
            retriever=self.retriever,
            llm=self.llm,
            max_generation_len=300,
            min_prob=0.3,
            verbose=True  # Enable debugging
        )
    # def setup_flare_chain(self):
    #     """
    #     Sets up the FLARE chain with correctly structured prompts.
    #     The key change is ensuring our prompts use 'user_input' as their variable name.
    #     """
    #     # Question generator prompt using 'user_input'
    #     question_prompt = ChatPromptTemplate.from_messages([
    #         ("system", f"""You are an expert at identifying unclear aspects in emails.
    #         Business Context: {self.business_context}
    #         Your task is to identify parts of the email that need clarification."""),
    #         ("user", "Review this email and identify uncertainties:\n{user_input}")  # Changed to user_input
    #     ])
        
    #     # Response generator prompt using 'user_input'
    #     response_prompt = ChatPromptTemplate.from_messages([
    #         ("system", f"""You are an expert at email classification.
    #         Business Context: {self.business_context}
    #         Classify the email into one of these categories: RECRUITMENT, INTERVIEW, ONBOARDING, GENERAL"""),
    #         ("user", "Classify this email:\n{user_input}")  # Changed to user_input
    #     ])
        
    #     # Create the chain components
    #     question_chain = question_prompt | self.lightweight_llm
    #     response_chain = response_prompt | self.llm
        
    #     # Initialize the FLARE chain
    #     self.flare_chain = FlareChain.invoke(
    #         question_prompt,
    #         llm=self.llm,
    #         retriever=self.retriever,
    #         min_prob=0.2,
    #         max_iter=3
    #     )

    async def process_email(self, email_data):
        """
        Processes a single email with the simplified input structure.
        Now we only need to provide email_content to match our prompt templates.
        """
        try:
            # Format the email content
            email_content = f"""
            Subject: {email_data.subject or ''}
            From: {email_data.sender_email or ''}
            To: {email_data.recipients or ''}
            Body: {email_data.body or ''}
            """
            
            # Call FLARE chain with the simplified input structure
            print("\n=== Processing Email ===")
            print(f"Email ID: {getattr(email_data, 'id', 'unknown')}")
            print("\nInput Content:")
            # print(email_content)
            
            # Call the FLARE chain and capture the result
            print("\nCalling FLARE chain...")
            result = self.flare_chain.run({
            "user_input": f"""
            Classify the emails under the following sales stages: {business_context} 
            Analyse similar emails to this and take a decision, email: {email_content}"""
            })

            
            # Detailed examination of the result
            print("\nFLARE Chain Result:")
            print(f"Result type: {type(result)}")
            print(f"Result content: {result}")
            
            if isinstance(result, dict):
                print("\nResult keys:", list(result.keys()))
                for key, value in result.items():
                    print(f"\nKey: {key}")
                    print(f"Value type: {type(value)}")
                    print(f"Value: {value}")
            
            print("\n=== End Processing ===\n")
            
            # For now, return a simple structure with the raw result
            return {
                "message_id": getattr(email_data, 'id', 'unknown'),
                "raw_result": result
            }
            
        except Exception as e:
            logging.error(f"Error processing email {email_data.id}: {str(e)}")
            raise

    async def process_batch(self, transaction, messages):
        processed_ids = []
        failed_ids = []
        
        for message in messages:
            try:
                result = await self.process_email(message)
                print(f"Successfully processed message {message.id}: {result['classification']}")
                processed_ids.append(message.id)
            except Exception as e:
                print(f"Failed to process message {message.id}: {str(e)}")
                failed_ids.append(message.id)
                
        return processed_ids, failed_ids

    async def process_all_messages(self):
        self.is_processing = True
        batch_size = 10
        start_from = 50
        
        try:
            async with self.prisma.tx(timeout=5000000) as transaction:
                total_messages = await transaction.messages.count()
                print(f"Found {total_messages} messages to process")
                
                processed_count = start_from
                failed_messages = []
                
                while processed_count < total_messages:
                    messages = await transaction.messages.find_many(
                        skip=processed_count,
                        take=batch_size,
                        order={'sent_date_time': 'desc'}
                    )
                    
                    if not messages:
                        break
                        
                    print(f"\nProcessing batch of {len(messages)} messages")
                    processed_ids, failed_ids = await self.process_batch(transaction, messages)
                    
                    processed_count += len(processed_ids)
                    failed_messages.extend(failed_ids)
                    
                    print(f"Progress: {(processed_count / total_messages) * 100:.1f}% ({processed_count}/{total_messages})")
                
                print(f"\nProcessing completed. Successful: {processed_count}, Failed: {len(failed_messages)}")
                
        except Exception as e:
            print(f"Fatal error during processing: {str(e)}")
            raise
        finally:
            self.is_processing = False