from src.agents.graph.workers import (
    claim_extraction_agent, ResourceClaimExtractionInputSchema,
    evidence_extraction_agent, EvidenceExtractionInputSchema
)
import json
from src.utils.pdf_parser import PDFParser
from src.agents.graph.workers import DebateKnowledgeGraph, ClaimNode, EvidenceNode

file = '/Users/stephen/Nottingham/arbitrary_arbitration/knowledge_source/quantitative_easing/qe_turbulance.pdf'
TOPIC = "Quantitative Easing is a good policy for long-term economic growth."
STANCE = "for"

def main():
    # Parse the PDF file
    pdf_parser = PDFParser()
    
    # Get the text from the PDF file
    raw_text = pdf_parser.pdf_to_text(file)

    # Extract claims from the text
    claim_res = claim_extraction_agent.run(
        ResourceClaimExtractionInputSchema(
            resource_raw_tex=raw_text,
            topic=TOPIC,
            stance=STANCE
        )
    )
    
    print(f"Claim: {claim_res}")

    rels = []
    ### build knowledge graph
    
    kg = DebateKnowledgeGraph()
    # extract evidence from the claims
    for claim in claim_res.claims:
        
        claim_node = kg.add_claim(claim)
        print(f"Claim node: {claim_node}")

        for_evidence_res = evidence_extraction_agent.run(
            EvidenceExtractionInputSchema(
                claim=claim,
                resource_raw_text=raw_text,
                stance='for',
            )
        )

        for evidence in for_evidence_res.evidence:
            rels.append({
                'claim': claim_node,
                'evidence': evidence,
                'is_support': True
            })
            print(f"added for evidence: {evidence}")

        against_evidence_res = evidence_extraction_agent.run(
            EvidenceExtractionInputSchema(
                claim=claim,
                resource_raw_text=raw_text,
                stance='against',
            )
        )

        for evidence in against_evidence_res.evidence:
            rels.append({
                'claim': claim_node,
                'evidence': evidence,
                'is_support': False
            })
            print(f"added against evidence: {evidence}")
    
    print("====== Relations ======")    

    for rel in rels:
        claim_node = rel['claim']
        evidence = rel['evidence']
        is_support = rel['is_support']

        kg.add_pair(claim_node, evidence, is_support)
    
    print("====== Knowledge Graph ======")
    
    ## save the graph
    with open('kg.json', 'w') as f:
        print (kg.to_json())
        json.dump(kg.to_json(), f, indent=4)

if __name__ == '__main__':
    main()
