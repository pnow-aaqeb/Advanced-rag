import json
import os
import logging
import re
import torch
import numpy as np
from typing import List, Tuple, Optional,Dict
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import html2text
from bge_singleton import BGEEmbeddings
from domain_anaylyzer import EmailDomainAnalyzer
from mongodb import mongodb
from email_parser import EmailContentProcessor
from prompts import business_context
# from email_processor import BGEEmbeddings
logger = logging.getLogger(__name__)

class ManualEmailClassifier:
    def __init__(self, prisma_client, pinecone_index, openai_client):
        self.prisma = prisma_client
        self.pinecone_index = pinecone_index
        self.openai = openai_client
        self.embeddings = BGEEmbeddings()
        self.categories = [
            "PROSPECT", "LEAD_GENERATION", "OPPORTUNITY",
            "FULFILLMENT", "DEAL", "SALE","OTHERS"
        ]
        self.confidence_threshold = 0.7
        
    async def _create_special_case_email_result(self, email, domain_analysis, category, confidence, 
                                         rationale, body_note="", skip=0, batch_size=10):
        """Create and save a standardized result for special case emails."""
        # Create the result object
        special_result = {
            "email_details": {
                "id": email.id,
                "subject": email.subject,
                "sender": email.sender_email,
                "recipients": email.recipients,
                "sent_date": email.sent_date_time.isoformat() if hasattr(email, 'sent_date_time') else None,
                "body": body_note or rationale
            },
            "domain_analysis": domain_analysis,
            "classification_process": {
                "final_result": {
                    "category": category,
                    "confidence": confidence,
                    "rationale": rationale
                }
            },
            "status": "success"
        }
        
        # Save to MongoDB if requested
        result_id = None
        if self.save_immediately:
            try:
                mongo_result = {
                    "status": "success",
                    "results": [special_result],
                    "total_processed": 1,
                    "next_skip": skip + batch_size
                }
                result_id = await mongodb.save_classification_result(mongo_result)
            except Exception as mongo_error:
                logger.error(f"MongoDB save error for email {email.id}: {str(mongo_error)}")
        
        # Update email category
        await self._update_email_category(email.id, category)
        
        return special_result, result_id
    
    async def classify_emails(self, batch_size=10, skip=100, save_immediately=True):
        """Main classification workflow"""
        try:
            emails = await self._get_unprocessed_emails(batch_size, skip)
            if not emails:
                return {"status": "no_emails", "message": "No unprocessed emails found"}
            
            classification_results = []
            saved_ids = []
            
            for email in emails:
                # Step 1: Process domain first to check for non-business domains
                domain_analyzer = EmailDomainAnalyzer()
                domain_analysis = domain_analyzer.analyze_email_addresses(
                    email.sender_email,
                    email.recipients
                )
                
                # Case 1: Non-business domain
                if domain_analysis.get('is_likely_non_business', False):
                    logger.info(f"Email using non-business domain detected: {domain_analysis['sender_domain']}")
                    result, result_id = await self._create_special_case_email_result(
                        email, domain_analysis, "OTHERS", 0.98,
                        f"Email uses non-business company domain ({domain_analysis['sender_domain']}) which is used for other purposes",
                        "NON-BUSINESS DOMAIN", skip, batch_size
                    )
                    classification_results.append(result)
                    if result_id:
                        saved_ids.append(result_id)
                    continue
                
                # Case 2: Empty email body
                if not email.body or email.body.isspace():
                    logger.warning(f"Empty email body detected for email ID: {email.id}")
                    result, result_id = await self._create_special_case_email_result(
                        email, domain_analysis, "OTHERS", 0.95,
                        "Email has empty body content",
                        "EMPTY", skip, batch_size
                    )
                    classification_results.append(result)
                    if result_id:
                        saved_ids.append(result_id)
                    continue
                
                # Process and clean email body
                email_processor = EmailContentProcessor()
                text_body, metadata = email_processor.extract_current_email(email.body)
                
                # Case 3: Empty after cleaning
                if not text_body.strip():
                    logger.warning(f"Email body empty after cleaning for email ID: {email.id}")
                    result, result_id = await self._create_special_case_email_result(
                        email, domain_analysis, "OTHERS", 0.95,
                        "Email has no meaningful content after cleaning",
                        "EMPTY AFTER CLEANING", skip, batch_size
                    )
                    classification_results.append(result)
                    if result_id:
                        saved_ids.append(result_id)
                    continue
                    logger.warning(f"Email body empty after cleaning for email ID: {email.id}")
                    empty_result = {
                        "email_details": {
                            "id": email.id,
                            "subject": email.subject,
                            "sender": email.sender_email,
                            "recipients": email.recipients,
                            "sent_date": email.sent_date_time.isoformat() if hasattr(email, 'sent_date_time') else None,
                            "body": "EMPTY AFTER CLEANING"
                        },
                        "classification_process": {
                            "final_result": {
                                "category": "OTHERS",
                                "confidence": 0.95,
                                "rationale": "Email has no meaningful content after cleaning"
                            }
                        },
                        "status": "success"
                    }
                    classification_results.append(empty_result)
                    
                    if save_immediately:
                        mongo_result = {
                            "status": "success",
                            "results": [empty_result],
                            "total_processed": 1,
                            "next_skip": skip + batch_size
                        }
                        result_id = await mongodb.save_classification_result(mongo_result)
                        saved_ids.append(result_id)
                    
                    await self._update_email_category(email.id, "OTHERS")
                    continue
                
                # Step 2: Initial classification with domain analysis
                domain_analyzer = EmailDomainAnalyzer() 
                domain_analysis = domain_analyzer.analyze_email_addresses(
                    email.sender_email,
                    email.recipients
                )
                
                # Log the final cleaned text and domain analysis
                logger.info(f"Cleaned email body: {text_body[:200]}...")
                logger.info(f"Domain analysis result: {domain_analysis}")
                
                initial_result = await self._initial_classification(email, domain_analysis, text_body)
                
                if 'uncertainty_points' in initial_result:
                    uncertainties = initial_result['uncertainty_points']
                elif 'final_result' in initial_result and 'uncertainty_points' in initial_result['final_result']:
                    uncertainties = initial_result['final_result']['uncertainty_points']
                else:
                    uncertainties = []

                # Step 3: Retrieve related context
                retrieved_context = await self._retrieve_related_context(
                    text_body,
                    email.subject,
                    email.sender_email,
                    email.recipients,
                    uncertainties
                )
                logger.info(f"Retrieved context: {retrieved_context}")
                
                # Step 4: Final classification with context
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
                
                # Save each email's result individually if requested
                if save_immediately:
                    try:
                        # Prepare single result for MongoDB
                        mongo_result = {
                            "status": "success",
                            "results": [email_result],
                            "total_processed": 1,
                            "next_skip": skip + batch_size  # This will be the same for all emails in batch
                        }
                        
                        # Save to MongoDB
                        result_id = await mongodb.save_classification_result(mongo_result)
                        saved_ids.append(result_id)
                        logger.info(f"Saved result for email {email.id} to MongoDB: {result_id}")
                    except Exception as mongo_error:
                        logger.error(f"MongoDB save error for email {email.id}: {str(mongo_error)}")
                
                classification_results.append(email_result)
                
                # Step 5: Update with final category
                await self._update_email_category(email.id, final_result["final_result"]["category"])
        
            if save_immediately:
                return {
                    "status": "success",
                    "result_ids": saved_ids,
                    "total_processed": len(saved_ids),
                    "message": "Classification completed and saved to database individually"
                }
            else:
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
        
    def create_special_case_result(self,category, confidence, rationale):
        """Create standardized result structure for special email cases."""
        classification = {
            "category": category,
            "confidence": confidence,
            "uncertainty_points": [],
            "rationale": rationale
        }
        
        return {
            "final_result": classification,
            "iterations": [{
                "iteration": 0,
                "classification": classification.copy(),
                "questions": None,
                "additional_context": None
            }],
            "total_iterations": 0
        }
    async def _initial_classification(self, email, domain_analysis, text_body) -> dict:
        """Get initial classification with uncertainty detection"""
        
        if domain_analysis.get('is_self_addressed', False):
            return self.create_special_case_result(
                "OTHERS", 0.95, 
                "Self-addressed email (likely template, draft, or test)"
            )
            
        #  Check for non-business domains
        if domain_analysis.get('is_likely_non_business', False):
            return self.create_special_case_result(
                "OTHERS", 0.98,
                f"Email uses non-business company domain ({domain_analysis['sender_domain']}) which is used for other purposes"
            )
    
        # Then check if this is likely a vendor email
        if domain_analysis.get('is_likely_vendor_email', False):
            return self.create_special_case_result(
                "OTHERS", 0.9,
                "Email identified as vendor communication (invoice/billing/subscription)"
            )
            
        #  Check for internal communication using personal emails
        if domain_analysis.get('is_internal_communication', False) and domain_analysis.get('sender_is_company_employee_personal_email', False):
            logger.info("Detected internal communication using personal email")
            
            # We'll still analyze the content, but with this additional context
            internal_note = "NOTE: This appears to be internal communication between company employees (personal email to company email)"
            
        else:
            internal_note = ""
            
        prompt = f"""You are an email classifier for ProficientNow, a recruitment company. Your task is to classify emails according to our sales pipeline stages while considering our detailed business context.
        
        IMPORTANT INSTRUCTIONS:
        1. Carefully distinguish between candidates ACCEPTING or DECLINING opportunities
        2. Look for clear indicators like "not interested", "decline", "accept", or "excited to join"
        3. An offer being extended is NOT the same as an offer being ACCEPTED
        4. Candidate rejections are FULFILLMENT stage, not SALE stage
        5. Only classify as SALE if there's explicit acceptance or confirmation of placement

        {internal_note}

        Email Domain Analysis:
        - Sender domain: {domain_analysis['sender_domain']}
        - Is likely candidate email: {domain_analysis['is_likely_candidate_email']}
        - Is internal communication: {domain_analysis.get('is_internal_communication', False)}
        - Using personal email mapped to company: {domain_analysis.get('sender_is_company_employee_personal_email', False)}
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
        Body: {text_body}

        Consider the following before classification:
        - Look for key indicators of the pipeline stage in the email content
        - Consider the sender and subject line context
        - Check if there are any stage-specific keywords or activities mentioned
        - Note any ambiguous phrases that could indicate multiple stages

        Return JSON format with:
        - category: (Select ONE stage from: PROSPECT, LEAD_GENERATION, OPPORTUNITY, FULFILLMENT, DEAL, SALE, OTHERS)
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

    async def retrieve_related_context(self, email_body, email_subject, sender_email, recipients, uncertainties=None):
        """Retrieve relevant context using field-specific embeddings with improved performance."""
        try:
            # Prepare recipient text
            recipients_text = ' '.join([r['emailAddress']['address'] for r in recipients 
                                    if 'emailAddress' in r and 'address' in r['emailAddress']])
            
            # Only search fields with content, prioritizing the most useful fields
            fields_to_search = []
            if email_body:
                fields_to_search.append(('body', email_body))
            if email_subject:
                fields_to_search.append(('subject', email_subject))
            if recipients_text:
                fields_to_search.append(('recipients', recipients_text))
            if sender_email:
                fields_to_search.append(('sender', sender_email))
                
            # Track seen emails to avoid duplicates
            seen_emails = set()
            results = []
            
            # Process each field (up to 3 most important fields for efficiency)
            for field_name, content in fields_to_search[:3]:
                try:
                    # Generate embedding and query
                    embedding = self.embeddings.embed_query(content)
                    query_response = self.pinecone_index.query(
                        namespace="email_embeddings",
                        vector=embedding,
                        top_k=5,  # Reduced from 10 for better performance
                        include_metadata=True
                    )
                    
                    # Process matches
                    for match in query_response.get('matches', []):
                        # Create a unique ID for deduplication
                        email_id = f"{match['metadata'].get('subject')}|{match['metadata'].get('sender_email')}"
                        if email_id in seen_emails:
                            continue
                            
                        seen_emails.add(email_id)
                        results.append({
                            'field': field_name,
                            'score': match.get('score', 0),
                            'metadata': match.get('metadata', {})
                        })
                except Exception as e:
                    logger.error(f"Error searching {field_name}: {str(e)}")
                    continue
                    
            # Sort by score and format results
            sorted_results = sorted(results, key=lambda x: -x['score'])[:10]
            formatted_results = [
                f"Similar email by {r['field']} - Subject: {r['metadata'].get('subject', 'No subject')} | "
                f"From: {r['metadata'].get('sender_email', 'No sender')} | "
                f"Date: {r['metadata'].get('sent_date', 'No date')} | "
                f"Score: {r['score']:.2f}"
                for r in sorted_results
            ]
            
            return "\n".join(formatted_results) if formatted_results else "No similar emails found."
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return "Error retrieving similar emails"


    async def _final_classification(self, email_body: str, context: str, initial_result: dict, domain_analysis) -> dict:
        """Make final classification with retrieved context"""

        if domain_analysis.get('is_self_addressed', False):
            # Wrap the result in the expected structure with iterations
            return {
                "final_result": {
                    "category": "OTHERS", 
                    "confidence": 0.95,
                    "uncertainty_points": [],
                    "rationale": "Self-addressed email (likely template, draft, or test)"
                },
                "iterations": [{
                    "iteration": 0,
                    "classification": {
                        "category": "OTHERS", 
                        "confidence": 0.95,
                        "uncertainty_points": [],
                        "rationale": "Self-addressed email (likely template, draft, or test)"
                    },
                    "questions": None,
                    "additional_context": None
                }],
                "total_iterations": 0
            }
            
        # NEW: Check for non-business domains
        if domain_analysis.get('is_likely_non_business', False):
            return {
                "final_result": {
                    "category": "OTHERS", 
                    "confidence": 0.98,
                    "uncertainty_points": [],
                    "rationale": f"Email uses non-business company domain ({domain_analysis['sender_domain']}) which is used for other purposes"
                },
                "iterations": [{
                    "iteration": 0,
                    "classification": {
                        "category": "OTHERS", 
                        "confidence": 0.98,
                        "uncertainty_points": [],
                        "rationale": f"Email uses non-business company domain ({domain_analysis['sender_domain']}) which is used for other purposes"
                    },
                    "questions": None,
                    "additional_context": None
                }],
                "total_iterations": 0
            }
    
        # Then check if this is likely a vendor email
        if domain_analysis['is_likely_vendor_email']:
            # Wrap the result in the expected structure with iterations
            return {
                "final_result": {
                    "category": "OTHERS",
                    "confidence": 0.9,
                    "uncertainty_points": [],
                    "rationale": "Email identified as vendor communication (invoice/billing/subscription)"
                },
                "iterations": [{
                    "iteration": 0,
                    "classification": {
                        "category": "OTHERS",
                        "confidence": 0.9,
                        "uncertainty_points": [],
                        "rationale": "Email identified as vendor communication (invoice/billing/subscription)"
                    },
                    "questions": None,
                    "additional_context": None
                }],
                "total_iterations": 0
            }
            
        # NEW: Special handling for internal communication using personal emails
        internal_comm_note = ""
        if domain_analysis.get('is_internal_communication', False) and domain_analysis.get('sender_is_company_employee_personal_email', False):
            internal_comm_note = """
            IMPORTANT: This appears to be internal communication between company employees where at least one person is using a personal email address. 
            Consider the content carefully to determine if this is:
            1. A test/template email (category=OTHERS)
            2. An actual business communication (category based on content)
            """
            
        base_prompt = f"""You are an expert email classifier for ProficientNow's recruitment pipeline. 
        Your task is to analyze this email and determine its category with high confidence.

        CRITICAL CLARIFICATION:
        The most common confusion is between LEAD_GENERATION and FULFILLMENT. Pay careful attention to:

        LEAD_GENERATION vs FULFILLMENT:
        - LEAD_GENERATION: Company talking TO CLIENT COMPANIES about general recruiting services
        * Offering to provide candidates (no specific candidates mentioned)
        * Marketing recruiting capabilities
        * General follow-up about staffing services

        - FULFILLMENT: Two key types of communications:
        * Communication WITH CANDIDATES about specific jobs:
            - Candidate inquiries about job details (even brief questions)
            - Company sending job descriptions to candidates
            - Candidate responses (interested or not interested)
        * Company talking TO CLIENTS about SPECIFIC candidates:
            - Sharing actual resumes/profiles
            - Discussing interview feedback for specific candidates
            - Coordinating interviews for specific candidates
            
        IMPORTANT: Check email direction carefully!
        - If from a generic email (gmail, etc.) TO company, and asking about job details = FULFILLMENT
        - Brief messages like "Can you tell me more about this role?" or "What are the benefits?" from a candidate = FULFILLMENT

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

        2. LEAD_GENERATION
        TARGET AUDIENCE: CLIENT COMPANIES (not candidates)
        DIRECTION: Company → Client Company
        PURPOSE: Marketing services, offering to fill positions
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
        TARGET AUDIENCE: CLIENT COMPANIES
        DIRECTION: Company ↔ Client Company
        PURPOSE: Gathering requirements, initial business discussions
        MUST HAVE:
        - Detailed requirement gathering from client
        - Business Development Manager involvement
        - Specific position requirements from client
        - Early discussions about potential business relationships
        - Initial negotiations about terms BEFORE a contract is drafted
        - Conversations about warranty periods and fee structures
        - Discussions that indicate interest but no firm commitment yet
        MUST NOT HAVE:
        - Candidate communication
        EXAMPLES:
        - "Please share your detailed requirements"
        - "Our BDM will contact you"

        4. FULFILLMENT
        TARGET AUDIENCE: CANDIDATES AND CLIENTS (regarding specific candidates)
        DIRECTION: 
        - Candidate ↔ Company (about jobs)
        - Company ↔ Client (about specific candidates)
        PURPOSE: Recruiting, interviewing, screening, profile sharing
        MUST HAVE AT LEAST ONE:
        - Direct candidate engagement with job details
        - Candidate inquiring about job specifics (even brief questions)
        - Sharing candidate profiles/resumes with clients
        - Interview coordination (schedules, feedback)
        - Candidate screening discussions
        - Client feedback on specific candidates
        COMMON CONTENT:
        - Job titles, compensation, and location details
        - Questions about job benefits, location, or type of work
        - Interview requests or feedback
        - Screening conversations
        - Candidate rejections or expressions of no interest
        - Discussion of candidate qualifications
        - "Here is the resume of [candidate name]"
        - Specific feedback about a candidate
        - Interview scheduling with clients
        MUST NOT HAVE:
        - Marketing general services (not specific candidates)
        - Initial service offering communications
        - Business development discussions without specific candidates
        - Contract discussions or payment terms
        EXAMPLES:
        - "I'd like to discuss this job opportunity with you"
        - "Here is the job description for the position we discussed"
        - "I'm not interested in this position" (from candidate)
        - "Can we schedule an interview?"
        - "Is this a remote position?" (from candidate)
        - "What benefits are offered?" (from candidate)
        - "Please find attached the resume for John Smith for your review"
        - "The candidate has 5 years of experience in Java development"
        - "Your feedback on the candidate we sent yesterday"

        5. DEAL
        TARGET AUDIENCE: CLIENT COMPANIES
        DIRECTION: Company ↔ Client Company
        PURPOSE: Finalizing agreements, contracts, terms
        MUST HAVE:
        - Final Contract discussions
        - Payment terms
        - Service agreements
        - Formal contract drafts being exchanged
        - Final negotiations on legal terms with intent to sign
        - Clear indication both parties have committed to work together
        - Specific contract language being discussed
        MUST NOT HAVE:
        - Job descriptions
        - Candidate communication
        EXAMPLES:
        - "Please review the final contract terms"
        - "Here are our payment milestones"

        6. SALE
        TARGET AUDIENCE: CLIENT COMPANIES
        DIRECTION: Company → Client Company
        PURPOSE: Confirming placements, invoicing, post-placement activities
        MUST HAVE at least ONE:
        - Confirmed candidate placements
        - Offer letters issued and accepted
        - Candidate joining confirmations
        - Invoice generation TO CLIENTS
        - Payment collection FROM CLIENTS
        - Post-placement follow-up
        MUST NOT HAVE:
        - Contract negotiations
        - Early stage discussions
        KEY INDICATORS:
        - Content: Placement confirmations, invoices to clients
        - Pattern: Post-deal operational communications
        - Intent: Managing active placements, collecting revenue
        EXAMPLES:
        - "The candidate has accepted and will join on March 1st"
        - "Please find attached the invoice for the placement"
        - "We've confirmed the start date for the candidate"
        - "Following up on our placed candidate's performance"

        7. OTHERS
        This includes:
        - Vendor invoices TO our company
        - Subscription notices FROM external services
        - Administrative emails unrelated to recruitment
        - Auto-generated notifications
        - Internal operations unrelated to specific deals
        EXAMPLES:
        - "Your subscription payment is due"
        - "Invoice from [Vendor] to ProficientNow"
        - "Office closure notification"

        IMPORTANT DECISION TREE FOR CLASSIFICATION:
        1. Is the email internal communication about potential clients? → PROSPECT
        2. Is the email TO a candidate about a job opportunity? → FULFILLMENT
        3. Is the email TO a client about SPECIFIC candidates (resumes, interviews, feedback)? → FULFILLMENT
        4. Is the email TO a client company offering general recruiting services? → LEAD_GENERATION
        5. Is the email TO/FROM a client gathering detailed requirements? → OPPORTUNITY
        6. Is the email about contract terms and negotiations? → DEAL
        7. Is the email about confirmed placements or invoicing clients? → SALE
        8. Is the email unrelated to the recruitment process? → OTHERS

        IMPORTANT FOR INVOICE EMAILS:
        - If WE are sending an invoice TO a CLIENT for our services = SALE
        - If a VENDOR is sending an invoice TO US for their services = OTHERS

        Based on similar emails and business context, re-evaluate the classification and return a JSON with:
        - category: (ONE of: PROSPECT, LEAD_GENERATION, OPPORTUNITY, FULFILLMENT, DEAL, SALE, OTHERS)
        - confidence: (0-1 score)
        - rationale: (Explain why this category best fits, referencing both business context and similar emails found)
        """

        response = await self.openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": base_prompt}],
        response_format={"type": "json_object"}
        )
    
        current_result = json.loads(response.choices[0].message.content)
        # logger.info(f"Initial final classification result: {current_result}")

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
            # logger.info(f"Generated questions for iteration {current_iteration + 1}: {questions}")
            
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
                        # logger.info(f"additional context:{results}")
                        
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
            
            COMMON MISCLASSIFICATION PATTERNS TO AVOID:
            1. DO NOT classify candidate rejections as SALE - these are FULFILLMENT
            2. DO NOT classify job descriptions sent to candidates as LEAD_GENERATION - these are FULFILLMENT
            3. DO NOT assume any email mentioning an offer is automatically SALE - check if it was ACCEPTED
            4. DO NOT classify candidate questions or concerns about a position as SALE - these are FULFILLMENT
            
            DOUBLE-CHECK THESE CRITICAL QUESTIONS:
            1. Is the message showing acceptance or rejection of an opportunity?
            2. Is a candidate declining interest or expressing interest?
            3. Are there explicit words like "not interested", "decline", or "pass" in the message?
            4. Is this communication BETWEEN a recruiter and a candidate (FULFILLMENT) or ABOUT a confirmed placement to a client (SALE)?
            5. Has the candidate formally accepted the offer and confirmed starting? Only then is it SALE.
            
            Based on ALL available information, provide your final classification as JSON with:
            - category: (ONE of: PROSPECT, LEAD_GENERATION, OPPORTUNITY, FULFILLMENT, DEAL, SALE, OTHERS)
            - confidence: (0-1 score)
            - rationale: (Detailed explanation including how the additional context influenced the decision)
            """
            
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": final_prompt}],
                response_format={"type": "json_object"}
            )
            
            current_result = json.loads(response.choices[0].message.content)
            # logger.info(f"Iteration {current_iteration + 1} result: {current_result}")

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

    async def _get_unprocessed_emails(self, batch_size: int, skip: int):
        """Retrieve unprocessed emails from database"""
        try:
            # No need to connect/disconnect here since it's managed in the task
            return await self.prisma.messages.find_many(
                skip=skip,
                take=batch_size,
                order={'sent_date_time': 'desc'}
            )
        except Exception as e:
            logger.error(f"Error fetching messages: {str(e)}")
            raise

    async def _update_email_category(self, email_id: str, category: str):
        """Update database with classification result"""
        # await self.prisma.email.update(
        #     where={"id": email_id},
        #     data={"category": category}
        # )
        print(f"category is {category}")
        