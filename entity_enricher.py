from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import logging

class EntityEnricher:
    def __init__(self,prisma_client,lighweigth_llm):
        self.prisma=prisma_client
        self.llm=lighweigth_llm
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at identifying business entities in emails.
            Focus only on identifying companies, candidates, positions, and contacts.
            Only return high-confidence matches (>0.8).
            
            Response format:
            {
                "entities": [
                    {
                        "type": "company",
                        "name": "exact name",
                        "confidence": confidence score,
                        "text": "extracted context"
                    }
                ]
            }"""),
            ("user", "Extract entities from this email:\n\n{email_content}")
        ])
        
        self.extraction_chain = LLMChain(
            llm=self.llm,
            prompt=self.extraction_prompt
        )

    async def extract_and_enrich(self, email_data: Dict) -> Tuple[List[ExtractedEntity], Dict]:
        """Extract entities and enrich with database information"""
        entities = await self.extract_entities(email_data)
        if not entities:
            return [], {}
            
        enriched_context = await self.enrich_entities(entities)
        return entities, enriched_context

    async def extract_entities(self, email_data: Dict) -> List[ExtractedEntity]:
        """Extract entities using lightweight LLM"""
        email_content = self.format_email_content(email_data)
        
        try:
            result = await self.extraction_chain.acall({"email_content": email_content})
            return [ExtractedEntity(**entity) for entity in result.get("entities", [])]
        except Exception as e:
            logging.error(f"Entity extraction failed: {e}")
            return []

    async def enrich_entities(self, entities: List[ExtractedEntity]) -> Dict:
        """Enrich entities with database information"""
        enriched = {
            "companies": [],
            "candidates": [],
            "positions": [],
            "contacts": []
        }
        
        for entity in entities:
            try:
                if entity.type == EntityType.COMPANY:
                    company = await self.get_company_info(entity.name)
                    if company:
                        enriched["companies"].append(company)
                elif entity.type == EntityType.CANDIDATE:
                    candidate = await self.get_candidate_info(entity.name)
                    if candidate:
                        enriched["candidates"].append(candidate)
                # Add similar methods for positions and contacts
            except Exception as e:
                logging.error(f"Enrichment failed for {entity.name}: {e}")
                
        return enriched

    async def get_company_info(self, name: str) -> Optional[Dict]:
        """Get detailed company information from database"""
        company = await self.prisma.companies.find_first(
            where={
                "OR": [
                    {"name": {"contains": name}},
                    {"domain": {"contains": name.lower()}}
                ]
            },
            include={
                "positions": True,
                "contacts": True
            }
        )
        
        if company:
            return {
                "id": company.id,
                "name": company.name,
                "domain": company.domain,
                "industry": company.industry,
                "active_positions": len([p for p in company.positions if p.is_active]),
                "key_contacts": [{"name": c.name, "role": c.job_title} for c in company.contacts]
            }
        return None