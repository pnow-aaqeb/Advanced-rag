business_context = """ ProficientNow Sales Pipeline Stages
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
        Fully Visible (through emails and system documentation) """

sales_stages = """
        CRITICAL CLARIFICATION:
        The most common confusion is between LEAD_GENERATION and FULFILLMENT. Pay careful attention to:

        LEAD_GENERATION vs FULFILLMENT:
        - LEAD_GENERATION: Company talking TO CLIENT COMPANIES about general recruiting services
            - Offering to provide candidates (no specific candidates mentioned)
            - Marketing recruiting capabilities
            - General follow-up about staffing services

        - FULFILLMENT: Two key types of communications:
            - Communication WITH CANDIDATES about specific jobs:
            - Candidate inquiries about job details (even brief questions)
            - Company sending job descriptions to candidates
            - Candidate responses (interested or not interested)
            - Company talking TO CLIENTS about SPECIFIC candidates:
            - Sharing actual resumes/profiles
            - Discussing interview feedback for specific candidates
            - Coordinating interviews for specific candidates
            
        IMPORTANT: Check email direction carefully!
        - If from a generic email (gmail, etc.) TO company, and asking about job details = FULFILLMENT
        - Brief messages like "Can you tell me more about this role?" or "What are the benefits?" from a candidate = FULFILLMENT
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
"""
question_context = """Consider these key aspects for each stage:

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
"""
