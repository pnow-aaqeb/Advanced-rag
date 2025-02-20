import json
import os
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple
from typing import List, Tuple, Optional

import numpy as np
from typing import List, Tuple, Optional
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import html2text

# from email_processor import BGEEmbeddings
logger = logging.getLogger(__name__)
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
        Detailed Client requirement gathering by the BDM
        Visibility:
        Fully Visible (emails visible)
        4. Fulfillment Stage
        Description:
        Assignment of recruiters
        Active candidate sourcing and contacting candidates
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
class EmailDomainAnalyzer:
    def __init__(self):
        # Known company domains (can be expanded)
        self.company_domains = {
            'proficientnow.com',
            # Add other known company domains here
        }
        
        # Common generic email domains
        self.generic_domains = {
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 
            'aol.com', 'icloud.com', 'protonmail.com', 'mail.com',
            'inbox.com', 'live.com', 'msn.com', 'ymail.com'
        }

    def analyze_email_addresses(self, sender_email: str, recipients: List[Dict]) -> Dict:
        """
        Analyzes email domains to help determine if this is likely a candidate communication.
        
        Args:
            sender_email: The email address of the sender
            recipients: List of recipient dictionaries with email addresses
            
        Returns:
            Dict containing analysis results
        """
        # Extract domains
        sender_domain = self._extract_domain(sender_email)
        recipient_domains = [self._extract_domain(r['emailAddress']['address']) 
                           for r in recipients if 'emailAddress' in r]
        
        # Analyze domain patterns
        analysis = {
            'sender_is_company': sender_domain in self.company_domains,
            'sender_is_generic': sender_domain in self.generic_domains,
            'sender_domain': sender_domain,
            'recipient_domains': list(set(recipient_domains)),
            'recipient_analysis': {
                'company_domains': len([d for d in recipient_domains if d in self.company_domains]),
                'generic_domains': len([d for d in recipient_domains if d in self.generic_domains]),
                'other_domains': len([d for d in recipient_domains 
                                    if d not in self.company_domains and d not in self.generic_domains])
            },
            'is_likely_candidate_email': False,
            'confidence': 0.0,
            'reasoning': []
        }
        
        # Add reasoning based on patterns
        if analysis['sender_is_company'] and analysis['recipient_analysis']['generic_domains'] > 0:
            analysis['reasoning'].append("Company sending to generic email addresses (typical for candidate communication)")
            analysis['is_likely_candidate_email'] = True
            analysis['confidence'] += 0.4
            
        if not analysis['sender_is_company'] and analysis['sender_is_generic']:
            analysis['reasoning'].append("Sender using generic email domain (typical for candidates)")
            analysis['is_likely_candidate_email'] = True
            analysis['confidence'] += 0.3
            
        if analysis['recipient_analysis']['company_domains'] > 0:
            analysis['reasoning'].append("Company domains in recipients (internal communication)")
            analysis['confidence'] += 0.2
            
        # Analyze job-related content indicators
        analysis['job_related_indicators'] = self._analyze_job_indicators(analysis)
        
        return analysis
    
    def _analyze_job_indicators(self, domain_analysis: Dict) -> Dict:
        """
        Analyzes additional indicators that this might be a job-related email.
        """
        return {
            'has_company_sender': domain_analysis['sender_is_company'],
            'has_generic_recipients': domain_analysis['recipient_analysis']['generic_domains'] > 0,
            'is_internal_communication': domain_analysis['recipient_analysis']['company_domains'] > 0
        }
    
    def _extract_domain(self, email: str) -> str:
        """Safely extracts domain from email address."""
        try:
            return email.split('@')[1].lower()
        except (IndexError, AttributeError):
            return ""
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

class ManualEmailClassifier:
    def __init__(self, prisma_client, pinecone_index, openai_client):
        self.prisma = prisma_client
        self.pinecone_index = pinecone_index
        self.openai = openai_client
        self.embeddings = BGEEmbeddings()
        self.categories = [
            "PROSPECT", "LEAD_GENERATION", "OPPORTUNITY",
            "FULFILLMENT", "DEAL", "SALE"
        ]
        
        # Example confidence threshold
        self.confidence_threshold = 0.7

    async def classify_emails(self, batch_size=10,skip=100):
        """Main classification workflow"""
        try:
            emails = await self._get_unprocessed_emails(batch_size,skip)
            if not emails:
                return {"status": "no_emails", "message": "No unprocessed emails found"}
            
            classification_results = []
            
            for email in emails:
                # Step 1: Initial classification
                domain_analyzer = EmailDomainAnalyzer() 
                domain_analysis = domain_analyzer.analyze_email_addresses(
                email.sender_email,
                email.recipients
                )
                # logger.info(f"Domain analysis result: {domain_analysis}")
                text_body=self._html_to_text(email.body)
                print(f"email body:{text_body}")
                initial_result = await self._initial_classification(email,domain_analysis)
                logger.info(f"Initial classification result: {initial_result}")
                
                # Step 2: Always do retrieval for additional context
                retrieved_context = await self._retrieve_related_context(
                    email.body,
                    email.subject,
                    email.sender_email,
                    email.recipients,
                    initial_result['uncertainty_points']
                )
                logger.info(f"Retrieved context: {retrieved_context}")
                
                # Step 3: Final classification with context
                final_result = await self._final_classification(
                    text_body, 
                    retrieved_context,
                    initial_result,
                    domain_analysis
                )
                logger.info(f"Final classification result: {final_result}")

                similar_emails = []
                for line in retrieved_context.split('\n'):
                    if line.startswith('Similar email'):
                        parts = line.split(' | ')
                        email_data = {}
                        for part in parts:
                            if ': ' in part:
                                key, value = part.split(': ', 1)
                                email_data[key.strip()] = value.strip()
                        if email_data:
                            similar_emails.append(email_data)

                email_result = {
                    "email_details": {
                        "id": email.id,
                        "subject": email.subject,
                        "sender": email.sender_email,
                        "recipients": email.recipients,
                        "sent_date": email.sent_date_time.isoformat() if hasattr(email, 'sent_date_time') else None,
                        "body": text_body
                    },
                    "domain_analysis": domain_analysis,
                    "initial_classification": initial_result,
                    "similar_emails": similar_emails,
                    "classification_process": {
                        "iterations": final_result["iterations"],
                        "total_iterations": final_result["total_iterations"],
                        "final_result": final_result["final_result"]
                    },
                    "status": "success"
                }
            
                classification_results.append(email_result)
                
                # Step 4: Update with final category
                await self._update_email_category(email.id, final_result["final_result"]["category"])
            return {
                "status": "success",
                "results": classification_results,
                "total_processed": len(classification_results)
            }
                    
        except Exception as e:
            logger.error(f"Error during email classification: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "results": []
            }

    async def _initial_classification(self, email,domain_analysis) -> dict:
        """Get initial classification with uncertainty detection"""
        prompt = f"""You are an email classifier for ProficientNow, a recruitment company. Your task is to classify emails according to our sales pipeline stages while considering our detailed business context.

        Email Domain Analysis:
        - Sender domain: {domain_analysis['sender_domain']}
        - Is likely candidate email: {domain_analysis['is_likely_candidate_email']}
        - Confidence: {domain_analysis['confidence']}
        - Reasoning: {', '.join(domain_analysis['reasoning'])}
        - Job related indicators: {domain_analysis['job_related_indicators']}

        Business Context:
        {business_context}

        Guidelines for classification:
        1. PROSPECT: Emails related to and internal documents researching potential clients 
        2. LEAD_GENERATION: Initial outreach emails, follow-ups, and basic information exchange, marketing emails
        3. OPPORTUNITY: Emails about detailed client requirements, communication with Business Development Manager
        4. FULFILLMENT: Emails about candidate sourcing, interviews, profile sharing, and interview coordination
        5. DEAL: Emails about contract negotiation, terms discussion, and legal documentation
        6. SALE: Emails about successful placements, offer letters, payments, and post-placement follow-up
        7. OTHERS: Auto generated or irrelevant emails


        Email to Classify:
        From: {email.sender_email}
        Subject: {email.subject}
        Body: {self._html_to_text(email.body)}

        Consider the following before classification:
        - Look for key indicators of the pipeline stage in the email content
        - Consider the sender and subject line context
        - Check if there are any stage-specific keywords or activities mentioned
        - Note any ambiguous phrases that could indicate multiple stages

        Return JSON format with:
        - category: (Select ONE stage from: PROSPECT, LEAD_GENERATION, OPPORTUNITY, FULFILLMENT, DEAL, SALE)
        - confidence: (0-1 score)
        - uncertainty_points: (list of ambiguous phrases that could affect classification)
        """
        
        response = await self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        logger.info(msg=f"response from intial classification:{response}")
        
        return json.loads(response.choices[0].message.content)

    async def _retrieve_related_context(self, email_body: str, email_subject: str, sender_email: str, recipients: str, uncertainties: List[str]) -> str:
        """Retrieve relevant context from vector store using all email fields"""
        context_pieces = []

        try:
            # Since recipients is already a list of dicts, no need for json.loads
            recipients_text = ' '.join([r['emailAddress']['address'] for r in recipients])
            logger.info(f"Processed recipients: {recipients_text}")
        except Exception as e:
            logger.error(f"Error processing recipients: {e}")
            recipients_text = ""

        # Dictionary of email fields to search
        email_fields = {
            'body': email_body,
            'subject': email_subject,
            'sender': sender_email,
            'recipients': recipients_text
        }

        # Search using each email field
        for field_name, field_content in email_fields.items():
            if not field_content:  # Skip empty fields
                logger.info(f"Skipping {field_name} search - empty content")
                continue
                
            try:
                field_embedding = self.embeddings.embed_query(field_content)
                embedding_values = field_embedding.tolist() if isinstance(field_embedding, np.ndarray) else field_embedding

                logger.info(f"{field_name.capitalize()} embedding dimension: {len(embedding_values)}")

                field_results = self.pinecone_index.query(
                    namespace="email_embeddings",
                    vector=embedding_values,
                    top_k=10,
                    include_metadata=True,
                    include_values=False
                )

                logger.info(f"{field_name.capitalize()} similarity search results: {field_results}")

                if field_results.get('matches'):
                    context_pieces.extend([
                        f"Similar email by {field_name} - Subject: {m['metadata'].get('subject', 'No subject')} | "
                        f"From: {m['metadata'].get('sender_email', 'No sender')} | "
                        f"Date: {m['metadata'].get('sent_date', 'No date')} | "
                        f"Score: {m.get('score', 0):.2f}"
                        for m in field_results['matches']
                    ])
            except Exception as e:
                logger.error(f"Error processing {field_name} search: {e}")
                continue


        # # Search using uncertainty points
        # for phrase in uncertainties:
        #     try:
        #         embedding = self.embeddings.embed_query(phrase)
        #         embedding_values = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                    
        #         logger.info(f"Uncertainty phrase: {phrase}")
        #         logger.info(f"Uncertainty embedding dimension: {len(embedding_values)}")

        #         results = self.pinecone_index.query(
        #             namespace="email_embeddings",
        #             vector=embedding_values,
        #             top_k=20,
        #             include_metadata=True,
        #             include_values=False
        #         )
                
        #         if results.get('matches'):
        #             context_pieces.extend([
        #                 f"Related email for '{phrase}' (category: {m['metadata'].get('category', 'unknown')}): {m['metadata'].get('snippet', 'no content')}"
        #                 for m in results['matches']
        #             ])
        #     except Exception as e:
        #         logger.error(f"Error processing uncertainty phrase '{phrase}': {e}")
        #         continue

        # Deduplicate results
        unique_pieces = []
        seen_snippets = set()
        
        for piece in sorted(context_pieces,key=lambda x:-float(x.split('Score: ')[1].strip(")"))):
            key = piece.split(": ", 1)[1]
            if key not in seen_snippets:
                seen_snippets.add(key)
                # print(f"seen:{seen_snippets}")
                unique_pieces.append(piece)
                # print(f"unique :{unique_pieces}")

        return "\n".join(unique_pieces[:10]) if unique_pieces else "No similar emails found."

    async def _final_classification(self, email_body: str, context: str, initial_result: dict,domain_analysis) -> dict:
        """Make final classification with retrieved context"""
        base_prompt = f"""You are an expert email classifier for ProficientNow's recruitment pipeline. 
        Your task is to analyze this email and determine its category with high confidence.

        Detailed Business Context:
        {business_context}

        Domain Analysis Results:
        {domain_analysis}

        Original Email Content:
        {email_body}

        Similar Emails Found:
        {context}

        Initial Assessment:
        - Category: {initial_result['category']}
        - Confidence: {initial_result['confidence']}
        - Uncertainty Points: {', '.join(initial_result['uncertainty_points'])}

        Classification Guidelines:

        Stage Definitions with STRICT Criteria:

        1. PROSPECT
        MUST HAVE:
        - Internal research about potential client companies
        - No direct client contact
        - Discussion of market research or company analysis
        MUST NOT HAVE:
        - Direct communication with clients/candidates
        - Job descriptions or requirements

        1. LEAD_GENERATION
        MUST HAVE at least ONE:
        - Offering to provide candidates to client companies
        - "Have candidates" or "qualified candidates" for client positions
        - Phrases like "would you like to review resumes" to clients
        - Follow-up about staffing services with clients
        - Outreach about having matches for client positions
        MUST NOT HAVE:
        - Direct candidate communication about applying to jobs
        - Sharing job descriptions with candidates
        - Interview scheduling with candidates
        KEY INDICATORS:
        - Content: Offering to share candidates/resumes with clients
        - Pattern: Recruiter to client company communication
        - Intent: Marketing available candidates to clients
        EXAMPLES:
        - "I have candidates for your position"
        - "Would you like to review the resumes"
        - "We have qualified matches for your role"
        3. OPPORTUNITY
        MUST HAVE:
        - Detailed requirement gathering from client
        - Business Development Manager involvement
        - Specific position requirements from client
        MUST NOT HAVE:
        - Candidate communication
        - Contract discussions
        EXAMPLES:
        - "Please share your detailed requirements"
        - "Our BDM will contact you"

        2. FULFILLMENT
        MUST HAVE ALL:
        - Company domain sending to generic email domain (gmail, yahoo, etc.)
        - Detailed job descriptions with qualifications
        - Direct candidate engagement
        MUST NOT HAVE:
        - Marketing messages to companies
        - Service offering communications
        - Business development discussions

        5. DEAL
        MUST HAVE:
        - Contract discussions
        - Payment terms
        - Service agreements
        MUST NOT HAVE:
        - Job descriptions
        - Candidate communication
        EXAMPLES:
        - "Please review the contract terms"
        - "Here are our payment milestones"

        6. SALE
        MUST HAVE:
        - Offer letters
        - Joining confirmations
        - Payment processing
        MUST NOT HAVE:
        - Initial job descriptions
        - Service discussions
        EXAMPLES:
        - "Congratulations on your offer"
        - "Please find your joining letter"
        7. OTHERS: Generic, Irrelevant, Auto-generated emails

        Based on similar emails and business context, re-evaluate the classification and return a JSON with:
        - category: (ONE of: PROSPECT, LEAD_GENERATION, OPPORTUNITY, FULFILLMENT, DEAL, SALE)
        - confidence: (0-1 score)
        - rationale: (Explain why this category best fits, referencing both business context and similar emails found)
        """

        response = await self.openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": base_prompt}],
        response_format={"type": "json_object"}
        )
    
        current_result = json.loads(response.choices[0].message.content)
        logger.info(f"Initial final classification result: {current_result}")

        iteration_results = [{
            "iteration": 0,
            "classification": current_result,
            "questions": None,
            "additional_context": None
        }]
        max_iterations = 3
        current_iteration = 0
        # current_result = initial_result
        
        while current_iteration < max_iterations and current_result['confidence'] < 0.9:
            # Generate questions about uncertain aspects
            question_prompt = f"""Based on the current classification:
            {json.dumps(current_result, indent=2)}
            
            Generate questions that would help confirm if this email belongs to a specific stage.

            Consider these key aspects for each stage:

            LEAD_GENERATION:
            - Direction: Company to client communication
            - Purpose: Marketing candidates/services 
            - Content pattern: Offering resources/candidates-
            - It could look like FULFILLMENT, but it can be a marketing gimic. 

            FULFILLMENT:
            - Direction: Company to candidate communication
            - Purpose: Recruitment activities
            - Content pattern: Job details, screening

            OPPORTUNITY:
            - Direction: Client requirement gathering
            - Purpose: Understanding needs
            - Content pattern: Specifications, planning

            DEAL:
            - Direction: Contract discussions
            - Purpose: Agreement finalization
            - Content pattern: Terms, conditions

            SALE:
            - Direction: Placement completion
            - Purpose: Onboarding, payment
            - Content pattern: Offers, joining

            Generate questions that would help clarify the true stage of this email.
            Focus on the email's:
            - Communication direction (who is talking to whom)
            - Purpose (what are they trying to achieve)
            - Content patterns (what kind of information is being shared)

            Generate targeted questions that would specifically help determine if the email matches the key criteria of each stage.
            These are the uncetainty points: {initial_result['uncertainty_points']}
            Format your response as a JSON list of questions.
            """
            
            questions_response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": question_prompt}],
                response_format={"type": "json_object"}
            )
            
            questions = json.loads(questions_response.choices[0].message.content)
            logger.info(f"Generated questions for iteration {current_iteration + 1}: {questions}")
            
            # Get additional context for each question
            additional_context = []
            # Fix the question extraction based on the response format
            stage_questions = []
            if 'LEAD_GENERATION_Questions' in questions:
                stage_questions.extend(questions['LEAD_GENERATION_Questions'])
            if 'FULFILLMENT_Questions' in questions:
                stage_questions.extend(questions['FULFILLMENT_Questions'])
            if 'OPPORTUNITY_Questions' in questions:
                stage_questions.extend(questions['OPPORTUNITY_Questions'])
            if 'DEAL_Questions' in questions:
                stage_questions.extend(questions['DEAL_Questions'])
            if 'SALE_Questions' in questions:
                stage_questions.extend(questions['SALE_Questions'])

            for question in stage_questions:
                if isinstance(question, str):  # Ensure question is a string
                    try:
                        embedding = self.embeddings.embed_query(question)
                        results = self.pinecone_index.query(
                            namespace="email_embeddings",
                            vector=embedding,
                            top_k=5,
                            include_metadata=True,
                            include_values=False
                        )
                        logger.info(f"additional context:{results}")
                        
                        if results.get('matches'):
                            additional_context.extend([
                                f"Context for '{question}': {m['metadata'].get('snippet', '')} (Category: {m['metadata'].get('category', 'unknown')})"
                                for m in results['matches']
                            ])
                    except Exception as e:
                        logger.error(f"Error processing question '{question}': {e}")
                        continue
            
            # Final classification with all context
            final_prompt = f"""{base_prompt}

            Additional Context from Iterative Analysis:
            {chr(10).join(additional_context)}

            Based on ALL available information, provide your final classification as JSON with:
            - category: (ONE of: PROSPECT, LEAD_GENERATION, OPPORTUNITY, FULFILLMENT, DEAL, SALE)
            - confidence: (0-1 score)
            - rationale: (Detailed explanation including how the additional context influenced the decision)
            """
            
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": final_prompt}],
                response_format={"type": "json_object"}
            )
            
            current_result = json.loads(response.choices[0].message.content)
            logger.info(f"Iteration {current_iteration + 1} result: {current_result}")

            iteration_results.append({
            "iteration": current_iteration + 1,
            "classification": current_result,
            "questions": questions,
            "additional_context": additional_context
            })
            
            current_iteration += 1
            
            # Break if confidence is high enough
            if current_result['confidence'] >= 0.9:
                break
        
        return {
        "final_result": current_result,
        "iterations": iteration_results,
        "total_iterations": current_iteration + 1
        }

    def _get_pipeline_descriptions(self) -> str:
        """Return formatted pipeline stages"""
        return """
        1. PROSPECT: Researching potential clients
        2. LEAD_GENERATION: Initial outreach and follow-ups
        3. OPPORTUNITY: Client requirements gathering
        4. FULFILLMENT: Candidate sourcing and interviews
        5. DEAL: Contract negotiations
        6. SALE: Successful placements and payments
        """.strip()
    def _html_to_text(self, body) -> str:
        """Clean and format email body text"""
        h = html2text.HTML2Text()
        h.ignore_images = True
        h.ignore_links = True
        h.ignore_tables = True
        
        # Convert HTML to text
        text = h.handle(body)
        
        # Clean up extra whitespace
        text = ' '.join(text.split())  # Remove multiple spaces
        
        # # Extract the most recent email (everything before the first "From:")
        # if "From:" in text:
        #     text = text.split("From:")[0].strip()
        
        # Additional cleaning if needed
        text = text.replace("**", "")  # Remove bold markers
        text = text.replace(">", "")   # Remove quote markers
        
        return text.strip()

    async def _get_unprocessed_emails(self, batch_size: int,skip:int):
        """Retrieve unprocessed emails from database"""
        return await self.prisma.messages.find_many(
            skip=skip,
            take=batch_size,
            order={'sent_date_time': 'desc'}
        )

    async def _update_email_category(self, email_id: str, category: str):
        """Update database with classification result"""
        # await self.prisma.email.update(
        #     where={"id": email_id},
        #     data={"category": category}
        # )
        print(f"category is {category}")