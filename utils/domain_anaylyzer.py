from typing import List,Dict
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
