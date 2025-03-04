import re
import html2text
import logging
from typing import Tuple, Dict, List, Any

logger = logging.getLogger(__name__)

class EmailContentProcessor:
    """
    Enhanced email parser optimized for HTML email content processing.
    Simplified to focus on core email extraction functionality.
    """
    
    @staticmethod
    def extract_current_email(email_content: str, handle_markdown: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Extracts only the current/most recent email body from a thread.
        Optimized for HTML content which is always expected.
        
        Args:
            email_content: The raw HTML email content
            handle_markdown: Whether to handle markdown symbols for display
            
        Returns:
            Tuple containing:
                - The extracted current email body text
                - Metadata dictionary with email details (sender, time, etc.)
        """
        if not email_content:
            logger.warning("Empty email content provided")
            return "", {"error": "Empty content"}
            
        # Convert HTML to text (always expected to be HTML)
        try:
            h = html2text.HTML2Text()
            h.ignore_images = True
            h.ignore_links = False
            h.ignore_tables = False
            h.body_width = 0  # Don't wrap text
            text_content = h.handle(email_content)
        except Exception as e:
            logger.error(f"Error converting HTML to text: {str(e)}")
            text_content = email_content  # Fallback to raw content
            
        # Common delimiter patterns that indicate the start of quoted/previous emails
        delimiter_patterns = [
            r'On\s+.*?,\s+.*?wrote:',                     
            r'On\s+.*?,\s+.*?<.*?>\s+wrote:',               
            r'On\s+.*?\s+at\s+.*?,\s+.*?wrote:',            
            r'From:[\s\S]*?Sent:[\s\S]*?To:[\s\S]*?Subject:', 
            r'From:\s*.*?\s*\[mailto:.*?\]',                
            r'-+\s*Original Message\s*-+',                  
            r'Begin forwarded message:',                     
            r'Forwarded message\s*-+',                       
            r'Reply all\s*Reply\s*Forward',                
            r'_+',                                          
            r'>{3,}',                                        
            r'From:.*?Subject:.*?\n\n',                     
            r'Sent from my iPhone',                          
            r'Sent from my Android',
            r'\*From:\*',
        ]
        
        # Create a single regex pattern from all delimiter patterns
        combined_pattern = '|'.join(f'({pattern})' for pattern in delimiter_patterns)
        
        # Find all matches
        try:
            matches = list(re.finditer(combined_pattern, text_content, re.MULTILINE | re.DOTALL))
        except Exception as e:
            logger.error(f"Error finding delimiter patterns: {str(e)}")
            matches = []
        
        # Extract metadata
        metadata = {}
        try:
            sender_match = re.search(r'From:\s*(.*?)(?=\s*<?[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}>?)', text_content)
            if sender_match:
                metadata['sender'] = sender_match.group(1).strip()
                
            email_match = re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text_content)
            if email_match:
                metadata['email'] = email_match.group(0)
                
            subject_match = re.search(r'Subject:\s*(.*?)(?=\n)', text_content)
            if subject_match:
                metadata['subject'] = subject_match.group(1).strip()
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
        
        # If no delimiter is found, this is likely a single email (not a thread)
        if not matches:
            # Clean up the text
            try:
                clean_text = re.sub(r'\n{3,}', '\n\n', text_content) 
                clean_text = re.sub(r'^\s+', '', clean_text, flags=re.MULTILINE)  
                
                # Process markdown if requested
                if handle_markdown:
                    clean_text = EmailContentProcessor._process_markdown_for_display(clean_text)
            except Exception as e:
                logger.error(f"Error cleaning text: {str(e)}")
                clean_text = text_content  # Fallback to original
            return clean_text.strip(), metadata
            
        # If delimiters are found, extract the content before the first one
        try:
            first_match = min(matches, key=lambda m: m.start())
            current_email = text_content[:first_match.start()].strip()
            
            # Post-processing to clean the extracted content
            # Remove header lines that might appear at the beginning of the current email
            current_email = re.sub(r'^(?:From|To|Cc|Subject|Date):.*\n', '', current_email, flags=re.MULTILINE)
            
            # Remove trailing signature markers
            signature_patterns = [
                r'--\s*$',                  
                r'Best regards,[\s\S]*$',    
                r'Regards,[\s\S]*$',         
                r'Thank you,[\s\S]*$',      
                r'Thanks,[\s\S]*$',         
                r'Sent from my (?:iPhone|iPad|Android|Galaxy|Samsung|mobile device)[\s\S]*$'  
            ]
            
            for pattern in signature_patterns:
                current_email = re.sub(pattern, '', current_email, flags=re.MULTILINE)
            
            # Clean up any remaining whitespace issues
            current_email = re.sub(r'\n{3,}', '\n\n', current_email)  # Normalize multiple newlines
            current_email = current_email.strip()
            
            # Process markdown if requested
            if handle_markdown:
                current_email = EmailContentProcessor._process_markdown_for_display(current_email)
        except Exception as e:
            logger.error(f"Error processing email content: {str(e)}")
            current_email = text_content[:1000] if len(text_content) > 1000 else text_content  # Fallback
        logger.info(msg=f"The is the clean current email:{current_email} and metadata:{metadata}")       
        return current_email, metadata
    
    @staticmethod
    def _process_markdown_for_display(text: str) -> str:
        """
        Process markdown formatting for display purposes.
        
        Args:
            text: The text to process
            
        Returns:
            Processed text with markdown handled appropriately
        """
        if not text:
            return ""
            
        try:
            # Handle bold text (** or __)
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
            text = re.sub(r'__(.*?)__', r'\1', text)
            
            # Handle italic text (* or _)
            text = re.sub(r'\*(.*?)\*', r'\1', text)  
            text = re.sub(r'_(.*?)_', r'\1', text)
            
            # Handle bullet points
            text = re.sub(r'^\s*\*\s', '• ', text, flags=re.MULTILINE)
            text = re.sub(r'^\s*-\s', '• ', text, flags=re.MULTILINE)   
            
            # Handle numbered lists
            text = re.sub(r'^\s*(\d+)\.\s', r'\1. ', text, flags=re.MULTILINE)
            
            # Handle links - extract just the link text
            text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
            
            return text
        except Exception as e:
            logger.error(f"Error processing markdown: {str(e)}")
            return text  # Return original if processing fails
        
    @staticmethod
    def is_empty_or_minimal_content(email_body: str) -> bool:
        """
        Checks if an email has empty or minimal content (like just a signature).
        
        Args:
            email_body: The email body text
            
        Returns:
            Boolean indicating if the email has substantive content
        """
        if not email_body or email_body.isspace():
            return True
            
        try:
            # Remove common signatures and links
            stripped_body = email_body
            signature_patterns = [
                r'Get \[Outlook for iOS\].*',
                r'Sent from my (?:iPhone|iPad|Android|Galaxy|Samsung|mobile device)',
                r'--\s*.*',
                r'Regards,.*$',
                r'Thank you,.*$',
                r'Best,.*$',
            ]
            for pattern in signature_patterns:
                stripped_body = re.sub(pattern, '', stripped_body, flags=re.DOTALL)
                
            # Log the stripped content for debugging
            logger.debug(f"Stripped body for analysis: {stripped_body[:100]}...")
                
            # More reliable word counting for structured emails
            words = re.findall(r'[A-Za-z0-9]+', stripped_body)
            
            # Count all words, not just "meaningful" ones
            total_words = len(words)
            
            # Count meaningful words (longer than 2 characters) as a backup check
            meaningful_words = [word for word in words if len(word) > 2]
            
            # Log the counts for debugging
            logger.debug(f"Total words: {total_words}, Meaningful words: {len(meaningful_words)}")
            
            # Consider email empty if both counts are very low
            return total_words < 10 and len(meaningful_words) < 3
        except Exception as e:
            logger.error(f"Error checking if content is empty: {str(e)}")
            return True  # Assume empty on error
    
    @staticmethod
    def prepare_email_for_embedding(email_data: Dict[str, Any]) -> str:
        """
        Prepare email for embedding by combining relevant fields with appropriate weighting.
        
        Args:
            email_data: Dictionary containing email fields
            
        Returns:
            Combined text optimized for embedding
        """
        try:
            parts = []
            
            # Add subject with higher importance
            if email_data.get('subject'):
                parts.append(f"SUBJECT: {email_data['subject']} {email_data['subject']}")  # Repeat for emphasis
                
            # Add sender
            if email_data.get('sender_email'):
                parts.append(f"FROM: {email_data['sender_email']}")
                
            # Add recipients (simplified)
            if email_data.get('recipients'):
                try:
                    recipient_emails = [r['emailAddress']['address'] for r in email_data['recipients'] 
                                      if 'emailAddress' in r and 'address' in r['emailAddress']]
                    if recipient_emails:
                        parts.append(f"TO: {', '.join(recipient_emails)}")
                except (KeyError, TypeError) as e:
                    logger.warning(f"Error processing recipients: {str(e)}")
            
            # Add body
            if email_data.get('body'):
                parts.append(email_data['body'])
                
            return "\n\n".join(parts)
        except Exception as e:
            logger.error(f"Error preparing email for embedding: {str(e)}")
            # Fallback to just using body if available
            return email_data.get('body', "")
    
    @staticmethod
    def get_email_parts(email_content: str, handle_markdown: bool = True) -> dict:
        """
        Extracts structured information from an HTML email.
        
        Args:
            email_content: The raw HTML email content
            handle_markdown: Whether to process markdown formatting
            
        Returns:
            Dictionary with email components:
                - body: The main body text
                - sender: Sender information
                - recipients: Recipient information
                - subject: Email subject
                - date: Email date/time
                - quoted_text: Any quoted text from previous emails
        """
        # First extract just the current email
        current_body, metadata = EmailContentProcessor.extract_current_email(email_content, handle_markdown)
        
        # Convert to text
        h = html2text.HTML2Text()
        h.ignore_images = True
        h.ignore_links = False
        h.body_width = 0
        full_text = h.handle(email_content)
            
        # Extract structured header information
        email_parts = {
            'body': current_body,
            'sender': None,
            'recipients': [],
            'subject': None,
            'date': None,
            'quoted_text': None
        }
        
        # Fill in metadata from the extraction
        email_parts.update({k: v for k, v in metadata.items() if v})
        
        # Try to get more structured header information
        headers = {
            'sender': r'From:\s*(.*?)(?=\n\S)',
            'recipients': r'To:\s*(.*?)(?=\n\S)',
            'subject': r'Subject:\s*(.*?)(?=\n\S)',
            'date': r'Date:\s*(.*?)(?=\n\S)'
        }
        
        for key, pattern in headers.items():
            match = re.search(pattern, full_text, re.DOTALL)
            if match and not email_parts.get(key):
                value = match.group(1).strip()
                # Special handling for recipients which might be a list
                if key == 'recipients' and ',' in value:
                    email_parts[key] = [r.strip() for r in value.split(',')]
                else:
                    email_parts[key] = value
                    
        # Extract quoted text (previous emails in the thread)
        current_body_pos = full_text.find(current_body) if current_body in full_text else 0
        if current_body_pos > 0:
            email_parts['quoted_text'] = full_text[:current_body_pos].strip()
        elif current_body_pos == 0 and len(current_body) < len(full_text):
            email_parts['quoted_text'] = full_text[len(current_body):].strip()
            
        return email_parts