# db/mongodb.py
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

class MongoDB:
    def __init__(self):
        self.client = None
        self.db = None
        self.results = None
    async def connect(self):
        """Establish connection to MongoDB"""
        self.client = AsyncIOMotorClient(os.getenv('MONGODB_URI'))
        self.db = self.client.email_classification
        self.results = self.db.classification_results
        # Test connection
        await self.db.command('ping')
    
    async def disconnect(self):
        """Close MongoDB connection"""
        self.client.close()

    async def save_classification_result(self, result: dict) -> str:
        """
        Save classification result using a structure that matches frontend requirements.
        The document will have top-level keys for email_details, domain_analysis,
        initial_classification, similar_emails, classification_process, and status.
        """
        # For convenience, we assume that result["results"] is an array with one primary result.
        primary = result.get("results", [{}])[0]

        document = {
            "timestamp": datetime.utcnow(),
            "status": result.get("status", "unknown"),
            "email_details": {
                "id": primary.get("email_details", {}).get("id"),
                "subject": primary.get("email_details", {}).get("subject"),
                "sender": primary.get("email_details", {}).get("sender"),
                "recipients": primary.get("email_details", {}).get("recipients", []),
                "sent_date": primary.get("email_details", {}).get("sent_date"),
                "body": primary.get("email_details", {}).get("body")
            },
            "domain_analysis": {
                "sender_is_company": primary.get("domain_analysis", {}).get("sender_is_company"),
                "sender_is_generic": primary.get("domain_analysis", {}).get("sender_is_generic"),
                "sender_domain": primary.get("domain_analysis", {}).get("sender_domain"),
                "recipient_domains": primary.get("domain_analysis", {}).get("recipient_domains", []),
                "recipient_analysis": {
                    "company_domains": primary.get("domain_analysis", {}).get("recipient_analysis", {}).get("company_domains", 0),
                    "generic_domains": primary.get("domain_analysis", {}).get("recipient_analysis", {}).get("generic_domains", 0),
                    "other_domains": primary.get("domain_analysis", {}).get("recipient_analysis", {}).get("other_domains", 0)
                },
                "is_likely_candidate_email": primary.get("domain_analysis", {}).get("is_likely_candidate_email"),
                "confidence": primary.get("domain_analysis", {}).get("confidence"),
                "reasoning": primary.get("domain_analysis", {}).get("reasoning", []),
                "job_related_indicators": primary.get("domain_analysis", {}).get("job_related_indicators", {})
            },
            "initial_classification": {
                "category": primary.get("initial_classification", {}).get("category"),
                "confidence": primary.get("initial_classification", {}).get("confidence"),
                "rationale": primary.get("initial_classification", {}).get("rationale"),
                "uncertainty_points": primary.get("initial_classification", {}).get("uncertainty_points", [])
            },
            "similar_emails": [
                {
                    "Subject": email.get("Subject"),
                    "From": email.get("From"),
                    "Date": email.get("Date"),
                    "Score": email.get("Score")
                }
                for email in primary.get("similar_emails", [])
            ],
            "classification_process": {
                "iterations": [
                    {
                        "iteration": iteration.get("iteration"),
                        "classification": {
                            "category": iteration.get("classification", {}).get("category"),
                            "confidence": iteration.get("classification", {}).get("confidence"),
                            "rationale": iteration.get("classification", {}).get("rationale"),
                            "uncertainty_points": iteration.get("classification", {}).get("uncertainty_points", [])
                        },
                        "questions": iteration.get("questions"),
                        "additional_context": iteration.get("additional_context")
                    }
                    for iteration in primary.get("classification_process", {}).get("iterations", [])
                ],
                "total_iterations": primary.get("classification_process", {}).get("total_iterations", 0),
                "final_result": {
                    "category": primary.get("classification_process", {}).get("final_result", {}).get("category"),
                    "confidence": primary.get("classification_process", {}).get("final_result", {}).get("confidence"),
                    "rationale": primary.get("classification_process", {}).get("final_result", {}).get("rationale")
                }
            },
            "total_processed": result.get("total_processed", 1),
            "next_skip": result.get("next_skip")
        }
        
        insert_result = await self.results.insert_one(document)
        return str(insert_result.inserted_id)

    async def get_classification_results(self, skip: int = 0, limit: int = 100):
        """
        Get paginated classification results.
        Handles MongoDB ObjectId serialization.
        """
        cursor = self.results.find({}).sort("timestamp", DESCENDING).skip(skip).limit(limit)
        results = await cursor.to_list()
        
        # Convert MongoDB ObjectId to string for JSON serialization
        for result in results:
            if '_id' in result:
                result['_id'] = str(result['_id'])
        
        return results

    async def get_total_results(self):
        """Get total count of classification results."""
        return await self.results.count_documents({})
        
    async def get_results_by_category(self, category: str, skip: int = 0, limit: int = 10):
        """
        Get results filtered by category.
        Handles MongoDB ObjectId serialization.
        """
        cursor = self.results.find({
            "classification_process.final_result.category": category
        }).sort("timestamp", DESCENDING).skip(skip).limit(limit)
        
        results = await cursor.to_list(length=limit)
        
        for result in results:
            if '_id' in result:
                result['_id'] = str(result['_id'])
        
        return results

# Initialize MongoDB instance
mongodb = MongoDB()
