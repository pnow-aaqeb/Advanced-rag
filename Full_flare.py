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
from mongodb import mongodb
from email_parser import EmailContentProcessor
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
            'proficientnowbooks.com',
            # Add other known company domains here
        }
        
        # Common generic email domains
        self.generic_domains = {
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 
            'aol.com', 'icloud.com', 'protonmail.com', 'mail.com',
            'inbox.com', 'live.com', 'msn.com', 'ymail.com','me.com'
        }

        
        self.non_business_company_domains = {
            'proficientnowbooks.com',  
            'proficientnowtech.com',   
        }

        self.vendor_domains = {
            'zohocorp.com',
            'microsoft.com',
            'adobe.com',
            'aws.amazon.com',
            'google.com',
            'salesforce.com',
            'slack.com',
            'dropbox.com',
            'atlassian.com',
            'github.com',
            'zendesk.com',
            'hubspot.com',
            'asana.com',
            'stripe.com',
            'docusign.com',
            'quickbooks.intuit.com'
            # Add more vendor domains as identified
        }

        self.employee_mappings = {
            'rmartin.pfn@icloud.com': 'rmartin@proficientnow.com',
        }

        self.employee_usernames = self._build_employee_usernames()
        
    def _build_employee_usernames(self) -> Dict[str, str]:
        """Build a mapping of usernames to full company emails for partial matching"""
        username_map = {}
        
        # Add mappings from employee_mappings
        for personal, company in self.employee_mappings.items():
            try:
                company_username = company.split('@')[0].lower()
                username_map[company_username] = company
            except (IndexError, AttributeError):
                continue
                
        
        return username_map
        
    def is_self_addressed_email(self, sender_email: str, recipients: List[Dict]) -> bool:
        """
        Checks if an email is sent from a person to themselves.
        This is likely a draft, template, or test email, not a real communication.
        """
        if not sender_email or not recipients:
            return False
            
        recipient_emails = [r['emailAddress']['address'].lower() 
                        for r in recipients if 'emailAddress' in r]
                        
        # Direct self-addressed check
        if sender_email.lower() in recipient_emails and len(recipient_emails) == 1:
            return True
            
        # NEW: Check for personal to company email or vice versa (same person)
        sender_lower = sender_email.lower()
        
        # Check if sender's personal email matches a recipient's company email
        if sender_lower in self.employee_mappings:
            company_email = self.employee_mappings[sender_lower]
            if company_email.lower() in recipient_emails and len(recipient_emails) == 1:
                return True
                
        # Check if sender's company email matches a recipient's personal email
        for personal, company in self.employee_mappings.items():
            if sender_lower == company.lower() and personal.lower() in recipient_emails and len(recipient_emails) == 1:
                return True
                
        # NEW: Check for username pattern matching
        try:
            sender_username = sender_lower.split('@')[0]
            
            # Check for username pattern in recipients
            for recipient in recipient_emails:
                try:
                    recipient_username = recipient.split('@')[0]
                    
                    # If usernames match and domains are different, likely same person
                    if sender_username == recipient_username and sender_lower != recipient:
                        return True
                except (IndexError, AttributeError):
                    continue
        except (IndexError, AttributeError):
            pass
            
        return False
    def is_likely_internal_communication(self, sender_email: str, recipients: List[Dict]) -> bool:
        """
        NEW: Check if this is likely communication between company employees,
        even if using personal emails.
        """
        if not sender_email or not recipients:
            return False
            
        sender_lower = sender_email.lower()
        recipient_emails = []
        for r in recipients:
            if 'emailAddress' in r:
                email = r['emailAddress']['address'].lower()
                recipient_emails.append(email)
        
        # Check if sender is using a known personal email for an employee
        sender_is_employee = False
        if sender_lower in self.employee_mappings:
            sender_is_employee = True
        elif any(d for d in self.company_domains if d in sender_lower):
            sender_is_employee = True
            
        # Also check if recipient includes company addresses or known employee personal emails
        recipient_has_employee = False
        for recipient in recipient_emails:
            # Direct company domain check
            if any(d for d in self.company_domains if d in recipient):
                recipient_has_employee = True
                break
                
            # Check for known personal emails of employees
            if recipient in self.employee_mappings:
                recipient_has_employee = True
                break
                
        # If both sender and at least one recipient are employees, it's internal
        return sender_is_employee and recipient_has_employee

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
        recipient_domains = []
        for r in recipients:
            if 'emailAddress' in r:
                email = r['emailAddress']['address']
                domain = self._extract_domain(email)
                recipient_domains.append(domain)
        
        # NEW: Check if sender is using a non-business company domain
        sender_is_non_business = sender_domain in self.non_business_company_domains
        
        # Initialize analysis dict FIRST before referencing it
        analysis = {
            'sender_is_company': sender_domain in self.company_domains,
            'sender_is_non_business_company_domain': sender_is_non_business,
            'sender_is_generic': sender_domain in self.generic_domains,
            'sender_is_vendor': sender_domain in self.vendor_domains,
            'sender_domain': sender_domain,
            'recipient_domains': list(set(recipient_domains)),
            'recipient_analysis': {
                'company_domains': len([d for d in recipient_domains if d in self.company_domains]),
                'non_business_domains': len([d for d in recipient_domains if d in self.non_business_company_domains]),
                'generic_domains': len([d for d in recipient_domains if d in self.generic_domains]),
                'vendor_domains': len([d for d in recipient_domains if d in self.vendor_domains]),
                'other_domains': len([d for d in recipient_domains 
                                    if d not in self.company_domains 
                                    and d not in self.non_business_company_domains
                                    and d not in self.generic_domains
                                    and d not in self.vendor_domains])
            },
            'email_direction': self._determine_email_direction(sender_domain, recipient_domains),
            'is_likely_candidate_email': False,
            'is_likely_vendor_email': False,
            'is_likely_non_business': sender_is_non_business or any(d in self.non_business_company_domains for d in recipient_domains),
            'is_self_addressed': self.is_self_addressed_email(sender_email, recipients),
            'is_internal_communication': self.is_likely_internal_communication(sender_email, recipients),
            'confidence': 0.0,
            'reasoning': []
        }
        
        # NEW: Special handling for non-business company domains
        if analysis['is_likely_non_business']:
            analysis['reasoning'].append("Email involves non-business company domain (should be classified as OTHERS)")
            analysis['confidence'] += 0.9
        
        # NEW: Check if personal email is being used by company employee
        if sender_email.lower() in self.employee_mappings:
            analysis['sender_is_company_employee_personal_email'] = True
            analysis['personal_to_company_mapping'] = self.employee_mappings[sender_email.lower()]
            analysis['reasoning'].append("Sender using known personal email that maps to company email")
        else:
            analysis['sender_is_company_employee_personal_email'] = False
            
        # Now add reasoning based on patterns
        if analysis['sender_is_vendor']:
            analysis['reasoning'].append("Email from a known vendor domain")
            analysis['is_likely_vendor_email'] = True
            analysis['confidence'] += 0.5
        
        if not analysis['sender_is_vendor'] and analysis['recipient_analysis']['vendor_domains'] > 0:
            analysis['reasoning'].append("Email to a known vendor domain")
            analysis['confidence'] += 0.3
            
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
            
        if analysis['is_self_addressed']:
            analysis['reasoning'].append("Email sent from a person to themselves (likely a template or test)")
            analysis['confidence'] += 0.8
            
        if analysis['is_internal_communication']:
            analysis['reasoning'].append("Communication between company employees (internal)")
            analysis['confidence'] += 0.6
            
        # Analyze job-related content indicators
        analysis['job_related_indicators'] = self._analyze_job_indicators(analysis)
        
        return analysis
    
    def _determine_email_direction(self, sender_domain: str, recipient_domains: List[str]) -> str:
        """
        Determine the direction of the email communication.
        
        Returns:
            str: "INBOUND" (to company), "OUTBOUND" (from company), "INTERNAL", or "EXTERNAL"
        """
        sender_is_company = sender_domain in self.company_domains
        recipients_include_company = any(domain in self.company_domains for domain in recipient_domains)
        if sender_domain in self.non_business_company_domains or any(domain in self.non_business_company_domains for domain in recipient_domains):
            return "NON_BUSINESS"
        if sender_is_company and not recipients_include_company:
            return "OUTBOUND"  # Company sending to outside
        elif not sender_is_company and recipients_include_company:
            return "INBOUND"   # Outside sending to company
        elif sender_is_company and recipients_include_company:
            return "INTERNAL"  # Within company
        else:
            return "EXTERNAL"  # Outside to outside (unusual case)
    
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
# class BGEEmbeddings(Embeddings):
#     def __init__(self, model_name: str = 'BAAI/bge-large-en-v1.5'):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.model = self.model.to(self.device)
        
#     def _get_embedding(self, text: str) -> List[float]:
#         inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            
#         norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
#         normalized_embeddings = embeddings / norm
#         return normalized_embeddings[0].tolist()

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return [self._get_embedding(text) for text in texts]

#     def embed_query(self, text: str) -> List[float]:
#         return self._get_embedding(text)

# class EmailContentProcessor:
#     """NEW: Class to handle email content processing and cleaning"""
    
#     @staticmethod
#     def clean_and_extract_recent_email(html_content: str) -> Tuple[str, bool]:
#         """
#         Extracts the most recent email from a thread and cleans the content.
        
#         Args:
#             html_content: Raw HTML email content
            
#         Returns:
#             Tuple containing:
#                 - Cleaned and extracted text
#                 - Boolean indicating if content is empty/invalid
#         """
#         if not html_content:
#             return "", True
            
#         # Convert HTML to text
#         h = html2text.HTML2Text()
#         h.ignore_images = True
#         h.ignore_links = False  # Keep links for better context
#         h.ignore_tables = False  # Keep tables for better structure
#         h.body_width = 0  # Don't wrap lines
        
#         text = h.handle(html_content)
        
#         # Check if content is still empty after conversion
#         if not text or text.isspace():
#             return "", True
            
#         # Try to extract just the most recent email from the thread
#         # Common patterns for quoted content/previous emails
#         patterns = [
#             r'From:.*?Subject:.*?\n\n',  # Standard email header pattern
#             r'On.*?wrote:',  # Common reply pattern
#             r'Begin forwarded message:',  # Forwarded message marker
#             r'-+Original Message-+',  # Another common separator
#             r'>.*',  # Quoted content (lines starting with >)
#             r'From: .*\[mailto:.*\]',  # Outlook-style headers
#             r'\*From:\*',  # Some HTML to text conversions
#             r'______+',  # Horizontal lines often separate messages
#             r'On\s+.*?,\s+.*?\s+wrote:',  # Gmail-style reply headers
#         ]
        
#         # Try to find the first occurrence of any pattern
#         positions = []
#         for pattern in patterns:
#             matches = list(re.finditer(pattern, text, re.MULTILINE | re.DOTALL))
#             for match in matches:
#                 positions.append(match.start())
        
#         # If we found separators, extract content before the first one
#         if positions:
#             first_pos = min(positions)
#             recent_content = text[:first_pos].strip()
            
#             # If we have content, return it
#             if recent_content:
#                 return recent_content, False
        
#         # Clean up the full text if we can't extract just the recent part
#         # Remove quoted lines (starting with >)
#         lines = text.splitlines()
#         clean_lines = [line for line in lines if not line.strip().startswith('>')]
        
#         # Rejoin and clean up whitespace
#         cleaned_text = '\n'.join(clean_lines)
#         cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Replace multiple newlines
#         cleaned_text = cleaned_text.strip()
        
#         return cleaned_text, not bool(cleaned_text)

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
        
        # Example confidence threshold
        self.confidence_threshold = 0.7
        

    
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
                
                # NEW: Check for non-business company domains first
                if domain_analysis.get('is_likely_non_business', False):
                    logger.info(f"Email using non-business domain detected: {domain_analysis['sender_domain']}")
                    non_business_result = {
                        "email_details": {
                            "id": email.id,
                            "subject": email.subject,
                            "sender": email.sender_email,
                            "recipients": email.recipients,
                            "sent_date": email.sent_date_time.isoformat() if hasattr(email, 'sent_date_time') else None,
                            "body": "NON-BUSINESS DOMAIN"
                        },
                        "domain_analysis": domain_analysis,
                        "classification_process": {
                            "final_result": {
                                "category": "OTHERS",
                                "confidence": 0.98,
                                "rationale": f"Email uses non-business company domain ({domain_analysis['sender_domain']}) which is used for other purposes"
                            }
                        },
                        "status": "success"
                    }
                    classification_results.append(non_business_result)
                    
                    # Save non-business email result if requested
                    if save_immediately:
                        mongo_result = {
                            "status": "success",
                            "results": [non_business_result],
                            "total_processed": 1,
                            "next_skip": skip + batch_size
                        }
                        result_id = await mongodb.save_classification_result(mongo_result)
                        saved_ids.append(result_id)
                    
                    # Update email category
                    await self._update_email_category(email.id, "OTHERS")
                    continue
                
                # NEW: Check if email body is empty
                if not email.body or email.body.isspace():
                    logger.warning(f"Empty email body detected for email ID: {email.id}")
                    empty_result = {
                        "email_details": {
                            "id": email.id,
                            "subject": email.subject,
                            "sender": email.sender_email,
                            "recipients": email.recipients,
                            "sent_date": email.sent_date_time.isoformat() if hasattr(email, 'sent_date_time') else None,
                            "body": "EMPTY"
                        },
                        "classification_process": {
                            "final_result": {
                                "category": "OTHERS",
                                "confidence": 0.95,
                                "rationale": "Email has empty body content"
                            }
                        },
                        "status": "success"
                    }
                    classification_results.append(empty_result)
                    
                    # Save empty email result if requested
                    if save_immediately:
                        mongo_result = {
                            "status": "success",
                            "results": [empty_result],
                            "total_processed": 1,
                            "next_skip": skip + batch_size
                        }
                        result_id = await mongodb.save_classification_result(mongo_result)
                        saved_ids.append(result_id)
                    
                    # Update email category
                    await self._update_email_category(email.id, "OTHERS")
                    continue
                
                # Step 1: Process and clean the email body
                email_processor = EmailContentProcessor()
                text_body, metadata = email_processor.extract_current_email(email.body)
                is_empty = not bool(text_body.strip())  
                
                # Handle empty content after cleaning
                if is_empty:
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
                
                # Step 2: Domain analysis
                domain_analyzer = EmailDomainAnalyzer() 
                domain_analysis = domain_analyzer.analyze_email_addresses(
                    email.sender_email,
                    email.recipients
                )

                # Log the final cleaned text and domain analysis
                logger.info(f"Cleaned email body: {text_body[:200]}...")
                logger.info(f"Domain analysis result: {domain_analysis}")

                # Step 3-4: Use FLARE for classification
                flare_result = await self._flare_classification(email, domain_analysis, text_body)
                logger.info(f"FLARE classification result: {flare_result}")

                # Extract values from the FLARE result
                all_retrieved_examples = flare_result.get("retrieved_examples", [])
                iterations = flare_result.get("iterations", [])
                final_classification = flare_result.get("final_result", {})

                # Format similar emails
                similar_emails = []
                for example in all_retrieved_examples:
                    similar_emails.append({
                        "Subject": example.get("subject", ""),
                        "From": example.get("sender", example.get("category", "")),
                        "Date": example.get("date", ""),
                        "Score": str(example.get("similarity", 0))
                    })

                # Format the result
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
                    "initial_classification": {
                        "category": iterations[0]["classification"].get("category", "") if iterations else "",
                        "confidence": iterations[0]["classification"].get("confidence", 0) if iterations else 0,
                        "uncertainty_points": iterations[0].get("questions", []) if iterations else []
                    },
                    "similar_emails": similar_emails,
                    "classification_process": {
                        "iterations": iterations,
                        "total_iterations": len(iterations),
                        "final_result": final_classification
                    },
                    "status": "success"
                }


                
                # Save each email's result individually if requested
                classification_results.append(email_result)

                # Save to MongoDB if requested
                if save_immediately:
                    try:
                        mongo_result = {
                            "status": "success",
                            "results": [email_result],
                            "total_processed": 1,
                            "next_skip": skip + batch_size
                        }
                        result_id = await mongodb.save_classification_result(mongo_result)
                        saved_ids.append(result_id)
                        logger.info(f"Saved result for email {email.id} to MongoDB: {result_id}")
                    except Exception as mongo_error:
                        logger.error(f"MongoDB save error for email {email.id}: {str(mongo_error)}")
                
                # Step 5: Update with final category
                await self._update_email_category(email.id, final_classification.get("category", "OTHERS"))
        
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
    async def _flare_classification(self, email, domain_analysis, text_body):
        """FLARE-based email classification with forward-looking active retrieval"""
        
        # Special case handling for fast-path behaviors
        if domain_analysis.get('is_self_addressed', False) or domain_analysis.get('is_likely_non_business', False) or domain_analysis['is_likely_vendor_email']:
            return self._handle_special_case(domain_analysis)
        
        # Initialize tracking variables
        iterations = []
        all_retrieved_examples = []
        max_iterations = 3
        reasoning_so_far = ""
        
        # Stage-specific confidence thresholds
        stage_confidence_thresholds = {
            "PROSPECT": 0.85,
            "LEAD_GENERATION": 0.88,  # Higher threshold as it's often confused with FULFILLMENT
            "OPPORTUNITY": 0.85,
            "FULFILLMENT": 0.88,      # Higher threshold as it's often confused with other stages
            "DEAL": 0.85,
            "SALE": 0.90,             # Highest threshold as misclassifying as SALE has business impact
            "OTHERS": 0.80            # Lower threshold as this is a catch-all category
        }
        
        # Track which aspects of the email are causing uncertainty
        uncertain_aspects = {
            "sender": False,
            "recipients": False,
            "subject": False, 
            "body": False
        }
        
        for i in range(max_iterations):
            # STEP 1: Generate a draft/look-ahead reasoning step
            draft_reasoning = await self._generate_look_ahead_step(
                email, text_body, domain_analysis, reasoning_so_far, i
            )
            
            # STEP 2: Identify uncertain parts in this draft step
            uncertain_parts = await self._identify_uncertainties(draft_reasoning, i)
            
            # Analyze which aspects of the email are causing uncertainty
            if uncertain_parts:
                uncertain_aspects = await self._analyze_uncertain_aspects(uncertain_parts, email)
            
            # If no uncertainties and we have a classification, we're done
            if not uncertain_parts and ("CLASSIFICATION:" in draft_reasoning or i == max_iterations - 1):
                # Formalize the classification at this step
                classification = await self._extract_classification_from_reasoning(
                    draft_reasoning, email, text_body, domain_analysis
                )
                
                # Check if we meet the stage-specific confidence threshold
                stage = classification.get("category", "OTHERS")
                threshold = stage_confidence_thresholds.get(stage, 0.85)
                
                if classification.get("confidence", 0) >= threshold:
                    # Add final iteration with sufficient confidence
                    iterations.append({
                        "iteration": i,
                        "classification": classification,
                        "questions": None,
                        "additional_context": None
                    })
                    break
            
            # STEP 3: For each uncertainty, perform targeted retrieval
            combined_context = ""
            if uncertain_parts:
                iteration_examples = []
                
                for part in uncertain_parts:
                    # Create targeted query based on which aspect is uncertain
                    query, query_aspect = await self._create_targeted_query(
                        part, email, text_body, uncertain_aspects
                    )
                    
                    # Perform targeted retrieval on the specific aspect's embeddings
                    query_examples = await self._targeted_retrieval(query, query_aspect)
                    
                    # Add to all examples
                    iteration_examples.extend(query_examples)
                    
                    # Format for reasoning refinement
                    for example in query_examples:
                        combined_context += f"Example ({query_aspect}) - Subject: {example.get('subject', '')}, " \
                                        f"Category: {example.get('category', '')}\n" \
                                        f"Snippet: {example.get('snippet', '')}\n\n"
                
                # Add unique examples to overall collection
                all_retrieved_examples.extend(iteration_examples)
                
                # STEP 4: Refine reasoning with retrieved information using the effective prompt
                refined_reasoning = await self._refine_with_retrieved_effective_prompt(
                    draft_reasoning, uncertain_parts, combined_context, 
                    email, text_body, domain_analysis
                )
                
                # Add to overall reasoning
                reasoning_so_far += "\n" + refined_reasoning
                
                # Extract classification if present
                classification = await self._extract_classification_from_reasoning(
                    refined_reasoning, email, text_body, domain_analysis
                )
                
                # Add iteration
                iterations.append({
                    "iteration": i,
                    "classification": classification,
                    "questions": [part["text"] for part in uncertain_parts],
                    "additional_context": combined_context,
                    "uncertain_aspects": uncertain_aspects
                })
                
                # If we got a classification with sufficient confidence, we're done
                if classification.get("category") and classification.get("confidence", 0) >= stage_confidence_thresholds.get(classification.get("category"), 0.85):
                    break
            else:
                # No uncertainties, just add to reasoning
                reasoning_so_far += "\n" + draft_reasoning
                
                # Add iteration without retrieval
                iterations.append({
                    "iteration": i,
                    "classification": {"reasoning": draft_reasoning},
                    "questions": None,
                    "additional_context": None
                })
        
        # Make final classification if we don't have one yet
        final_iteration = iterations[-1]
        if "category" not in final_iteration["classification"]:
            final_classification = await self._make_final_classification_effective_prompt(
                email, text_body, reasoning_so_far, domain_analysis, all_retrieved_examples
            )
        else:
            final_classification = final_iteration["classification"]
        
        # Return results in expected format
        return {
            "final_result": final_classification,
            "iterations": iterations,
            "total_iterations": len(iterations),
            "retrieved_examples": all_retrieved_examples
        }


    async def _generate_look_ahead_step(self, email, text_body, domain_analysis, reasoning_so_far, iteration):
        """Generate the next reasoning step using look-ahead approach"""
        prompt = f"""
        You are analyzing an email to classify it into ProficientNow's sales pipeline stages.
        
        Email content: 
        {text_body[:800]}...
        
        Subject: {email.subject}
        From: {email.sender_email}
        
        Domain analysis: {json.dumps(domain_analysis, indent=2)}
        
        Business context:
        {business_context}
        
        Reasoning steps taken so far:
        {reasoning_so_far}
        
        Generate the next step in analyzing this email. If uncertain about any aspect, 
        mark it with [UNCERTAIN: reason].
        
        Example: "This appears to be about candidate screening [UNCERTAIN: unclear if for an interview or initial contact]"
        
        If you have enough information to make a classification, include:
        CLASSIFICATION: [category] - [brief rationale]
        
        Keep your response focused on ONE specific aspect that needs analysis.
        """
        
        response = await self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        logger.info(f"this is the look ahead step:{response.choices[0].message.content}")
        
        return response.choices[0].message.content

    async def _identify_uncertainties(self, reasoning_step, iteration):
        """Extract uncertain parts from reasoning"""
        uncertain_parts = []
        
        # Find all parts marked with [UNCERTAIN: reason]
        pattern = r'(.*?)\s*\[UNCERTAIN:\s*(.*?)\]'
        matches = re.finditer(pattern, reasoning_step)
        
        for match in matches:
            uncertain_parts.append({
                "text": match.group(1).strip(),
                "reason": match.group(2).strip(),
                "full_match": match.group(0)
            })
        logger.info(f"these are the uncertainties:{uncertain_parts}")
        return uncertain_parts

    async def _create_targeted_query(self, uncertain_part, email, text_body, uncertain_aspects):
        """Create specific search query from uncertain part"""
        
        # Extract the base query from the uncertain part
        query_text = f"{uncertain_part['text']} {uncertain_part['reason']}"
        
        # Determine which aspect to focus on
        aspect = "body"  # Default
        for aspect_name, is_uncertain in uncertain_aspects.items():
            if is_uncertain:
                aspect = aspect_name
                break
        
        # Add context from the email based on the aspect
        if aspect == "sender":
            query_text += f" {email.sender_email}"
        elif aspect == "recipients":
            recipients_text = ' '.join([r['emailAddress']['address'] for r in email.recipients])
            query_text += f" {recipients_text}"
        elif aspect == "subject":
            query_text += f" {email.subject}"
        
        return query_text, aspect

    async def _targeted_retrieval(self, query, aspect):
        """Retrieve information targeting the specific uncertain aspect using proven approach"""
        try:
            # Generate embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Query Pinecone using the single namespace approach that was working
            results = self.pinecone_index.query(
                namespace="email_embeddings",
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            
            # Format results using your existing format
            formatted_results = []
            if results.get('matches'):
                formatted_results = [
                    {
                        "snippet": f"Subject: {m['metadata'].get('subject', 'No subject')} | "
                                f"From: {m['metadata'].get('sender_email', 'No sender')} | "
                                f"Date: {m['metadata'].get('sent_date', 'No date')} | "
                                f"Category: {m['metadata'].get('category', 'Unknown')}",
                        "subject": m['metadata'].get('subject', ''),
                        "sender": m['metadata'].get('sender_email', ''),
                        "category": m['metadata'].get('category', ''),
                        "similarity": m.get('score', 0),
                        "aspect": aspect
                    }
                    for m in results['matches']
                ]
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error in targeted retrieval: {str(e)}")
            return []

    async def _refine_with_retrieved_effective_prompt(self, draft_reasoning, uncertain_parts, retrieved_context, 
                                              email, text_body, domain_analysis):
        """Refine reasoning with retrieved information using the effective prompt"""
        
        # Use the effective prompt format that worked well in your first approach
        prompt = f"""You are an expert email classifier for ProficientNow's recruitment pipeline. 
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
        {text_body}

        Current Reasoning:
        {draft_reasoning}

        Uncertain aspects identified:
        {', '.join([part["full_match"] for part in uncertain_parts])}

        Retrieved Similar Examples:
        {retrieved_context}

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


        IMPORTANT FOR INVOICE EMAILS:
        - If WE are sending an invoice TO a CLIENT for our services = SALE
        - If a VENDOR is sending an invoice TO US for their services = OTHERS

        Rewrite your reasoning step to resolve the uncertainties based on the retrieved information.
        Replace [UNCERTAIN: reason] markers with confident statements.
        
        If you can now make a classification, include:
        CLASSIFICATION: [category] - [brief rationale]
        """
        
        response = await self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content

    async def _extract_classification_from_reasoning(self, reasoning, email, text_body, domain_analysis):
        """Extract classification if present in reasoning"""
        match = re.search(r'CLASSIFICATION:\s*(\w+)\s*-\s*(.*)', reasoning)
        if match:
            category = match.group(1).strip().upper()
            rationale = match.group(2).strip()
            
            # Validate category
            if category in self.categories:
                return {
                    "category": category,
                    "confidence": 0.9,  # High confidence since we resolved uncertainties
                    "rationale": rationale
                }
        
        # No valid classification found
        return {
            "reasoning": reasoning
        }

    async def _make_final_classification_effective_prompt(self, email, text_body, reasoning_so_far, domain_analysis, retrieved_examples):
        """Make final classification using the effective prompt when reasoning doesn't contain one"""
        
        # Format retrieved examples
        context = ""
        for example in retrieved_examples[:5]:  # Use top 5 examples
            context += f"Example - Subject: {example.get('subject', '')}, " \
                    f"Category: {example.get('category', '')}\n" \
                    f"Snippet: {example.get('snippet', '')}\n\n"
        
        # Use the effective base prompt
        prompt = f"""You are an expert email classifier for ProficientNow's recruitment pipeline. 
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
        {text_body}

        All Reasoning Steps So Far:
        {reasoning_so_far}

        Similar Emails Found:
        {context}

        Stage Definitions with STRICT Criteria:
        [full stage definitions from the base prompt...]

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

        Return a JSON with:
        - category: (ONE of: PROSPECT, LEAD_GENERATION, OPPORTUNITY, FULFILLMENT, DEAL, SALE, OTHERS)
        - confidence: (0-1 score)
        - rationale: (Explain why this category best fits, referencing both business context and similar emails found)
        """
        
        response = await self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    async def _analyze_uncertain_aspects(self, uncertain_parts, email):
        """Determine which aspects of the email are causing uncertainty"""
        uncertain_aspects = {
            "sender": False,
            "recipients": False,
            "subject": False, 
            "body": False
        }
        
        # Keywords that might indicate uncertainty about particular aspects
        sender_keywords = ["sender", "from", "who sent", "company domain", "email address"]
        recipient_keywords = ["recipient", "to whom", "who received", "addressed to"]
        subject_keywords = ["subject", "title", "topic", "heading"]
        
        # Check each uncertain part for keywords
        for part in uncertain_parts:
            uncertain_text = part["text"].lower() + " " + part["reason"].lower()
            
            if any(keyword in uncertain_text for keyword in sender_keywords):
                uncertain_aspects["sender"] = True
                
            if any(keyword in uncertain_text for keyword in recipient_keywords):
                uncertain_aspects["recipients"] = True
                
            if any(keyword in uncertain_text for keyword in subject_keywords):
                uncertain_aspects["subject"] = True
            
            # Body is the default if other aspects aren't specifically mentioned
            if not any(uncertain_aspects.values()):
                uncertain_aspects["body"] = True
    
        return uncertain_aspects
    def _handle_special_case(self, domain_analysis):
        """Handle special cases where detailed FLARE isn't needed"""
        category = "OTHERS"
        confidence = 0.95
        rationale = "Email identified as special case"
        
        if domain_analysis.get('is_self_addressed', False):
            rationale = "Self-addressed email (likely template, draft, or test)"
        elif domain_analysis.get('is_likely_non_business', False):
            confidence = 0.98
            rationale = f"Email uses non-business company domain ({domain_analysis['sender_domain']}) which is used for other purposes"
        elif domain_analysis['is_likely_vendor_email']:
            confidence = 0.9
            rationale = "Email identified as vendor communication (invoice/billing/subscription)"
        
        return {
            "final_result": {
                "category": category,
                "confidence": confidence,
                "rationale": rationale
            },
            "iterations": [{
                "iteration": 0,
                "classification": {
                    "category": category,
                    "confidence": confidence,
                    "rationale": rationale
                },
                "questions": None,
                "additional_context": None
            }],
            "total_iterations": 1,
            "retrieved_examples": []
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