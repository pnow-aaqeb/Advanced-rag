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

class EmailContentProcessor:
    """NEW: Class to handle email content processing and cleaning"""
    
    @staticmethod
    def clean_and_extract_recent_email(html_content: str) -> Tuple[str, bool]:
        """
        Extracts the most recent email from a thread and cleans the content.
        
        Args:
            html_content: Raw HTML email content
            
        Returns:
            Tuple containing:
                - Cleaned and extracted text
                - Boolean indicating if content is empty/invalid
        """
        if not html_content:
            return "", True
            
        # Convert HTML to text
        h = html2text.HTML2Text()
        h.ignore_images = True
        h.ignore_links = False  # Keep links for better context
        h.ignore_tables = False  # Keep tables for better structure
        h.body_width = 0  # Don't wrap lines
        
        text = h.handle(html_content)
        
        # Check if content is still empty after conversion
        if not text or text.isspace():
            return "", True
            
        # Try to extract just the most recent email from the thread
        # Common patterns for quoted content/previous emails
        patterns = [
            r'From:.*?Subject:.*?\n\n',  # Standard email header pattern
            r'On.*?wrote:',  # Common reply pattern
            r'Begin forwarded message:',  # Forwarded message marker
            r'-+Original Message-+',  # Another common separator
            r'>.*',  # Quoted content (lines starting with >)
            r'From: .*\[mailto:.*\]',  # Outlook-style headers
            r'\*From:\*',  # Some HTML to text conversions
            r'______+',  # Horizontal lines often separate messages
            r'On\s+.*?,\s+.*?\s+wrote:',  # Gmail-style reply headers
        ]
        
        # Try to find the first occurrence of any pattern
        positions = []
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE | re.DOTALL))
            for match in matches:
                positions.append(match.start())
        
        # If we found separators, extract content before the first one
        if positions:
            first_pos = min(positions)
            recent_content = text[:first_pos].strip()
            
            # If we have content, return it
            if recent_content:
                return recent_content, False
        
        # Clean up the full text if we can't extract just the recent part
        # Remove quoted lines (starting with >)
        lines = text.splitlines()
        clean_lines = [line for line in lines if not line.strip().startswith('>')]
        
        # Rejoin and clean up whitespace
        cleaned_text = '\n'.join(clean_lines)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Replace multiple newlines
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text, not bool(cleaned_text)

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
                text_body, is_empty = email_processor.clean_and_extract_recent_email(email.body)
                
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
    async def _initial_classification(self, email, domain_analysis, text_body) -> dict:
        """Get initial classification with uncertainty detection"""
        
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
            
        # NEW: Check for internal communication using personal emails
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

    async def _retrieve_related_context(self, email_body: str, email_subject: str, 
                                    sender_email: str, recipients: str, 
                                    uncertainties: List[str]) -> str:
        """Retrieve relevant context from vector store using all email fields"""
        context_pieces = []
        
        try:
            recipients_text = ' '.join([r['emailAddress']['address'] for r in recipients])
            # logger.info(f"Processing recipients: {recipients_text}")

            # Dictionary of email fields to search
            email_fields = {
                'body': email_body,
                'subject': email_subject,
                'sender': sender_email,
                'recipients': recipients_text
            }

            # Search using each email field
            for field_name, field_content in email_fields.items():
                if not field_content:
                    logger.info(f"Skipping {field_name} search - empty content")
                    continue
                    
                try:
                    # Generate embedding
                    field_embedding = self.embeddings.embed_query(field_content)
                    
                    # Log embedding details for debugging
                    logger.info(f"{field_name} embedding generated successfully. "
                            f"Dimension: {len(field_embedding)}")

                    # Query Pinecone
                    field_results = self.pinecone_index.query(
                        namespace="email_embeddings",
                        vector=field_embedding,
                        top_k=10,
                        include_metadata=True
                    )

                    # Log query results
                    logger.info(f"{field_name} search returned "
                            f"{len(field_results.get('matches', []))} matches")

                    if field_results.get('matches'):
                        context_pieces.extend([
                            f"Similar email by {field_name} - Subject: {m['metadata'].get('subject', 'No subject')} | "
                            f"From: {m['metadata'].get('sender_email', 'No sender')} | "
                            f"Date: {m['metadata'].get('sent_date', 'No date')} | "
                            f"Score: {m.get('score', 0):.2f}"
                            for m in field_results['matches']
                        ])
                except Exception as e:
                    logger.error(f"Error in {field_name} search: {str(e)}", exc_info=True)
                    continue

            # Deduplicate and sort results
            unique_pieces = []
            seen_snippets = set()
            
            for piece in sorted(context_pieces, 
                            key=lambda x: -float(x.split('Score: ')[1].strip(")"))):
                key = piece.split(": ", 1)[1]
                if key not in seen_snippets:
                    seen_snippets.add(key)
                    unique_pieces.append(piece)

            return "\n".join(unique_pieces[:10]) if unique_pieces else "No similar emails found."
        
        except Exception as e:
            logger.error(f"Error in _retrieve_related_context: {str(e)}", exc_info=True)
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

        {internal_comm_note}

        CRITICAL INSTRUCTIONS:
        1. Carefully distinguish between ACCEPTANCE and REJECTION communications
        2. Pay special attention to phrases like "not interested", "decline", "reject" - these indicate FULFILLMENT stage
        3. Only classify as SALE if there is explicit confirmation of ACCEPTANCE, JOINING or PAYMENT
        4. A job offer being extended is NOT the same as an offer being ACCEPTED
        5. Verify the actual outcome (acceptance vs rejection) before making your decision

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
        - Early discussions about potential business relationships
        - Initial negotiations about terms BEFORE a contract is drafted
        - Conversations about warranty periods and fee structures
        - Discussions that indicate interest but no firm commitment yet
        MUST NOT HAVE:
        - Candidate communication
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