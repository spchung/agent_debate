from src.agents.graph.workers import (
    claim_extraction_agent, ResourceClaimExtractionInputSchema,
    evidence_extraction_agent, EvidenceExtractionInputSchema
)
import json
from src.utils.pdf_parser import PDFParser
from src.agents.graph.workers import DebateKnowledgeGraph, ClaimNode, EvidenceNode
from src.utils.in_mem_vector_store import InMemoryVectorStore

file = '/Users/stephen/Nottingham/arbitrary_arbitration/knowledge_source/quantitative_easing/qe_turbulance.pdf'
TOPIC = "Quantitative Easing is a good policy for long-term economic growth."
STANCE = "for"

def step1():
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
    
    for rel in rels:
        claim_node = rel['claim']
        evidence_node = EvidenceNode(rel['evidence'])
        is_support = rel['is_support']

        kg.add_pair(claim_node, evidence_node, is_support)
    
    ## save the graph
    with open('kg.json', 'w') as f:
        print (kg.to_json())
        json.dump(kg.to_json(), f, indent=4)

def step2():
    ### inference

    # read form the json file
    data = None
    with open('kg.json', 'r') as f:
        data = json.load(f)
    
    # create a new instance of DebateKnowledgeGraph
    kg = DebateKnowledgeGraph.from_dict(data)

    opponent_response = 'QE is an ineffective method of stimulating the economy. It leads to asset bubbles and income inequality.'

    ## find the closest claim in the graph
    claim = kg.query_claim(opponent_response)

    ## get related evidence
    supporting_evdience = []
    refuting_evidence = []

    if claim:
        print(f"Claim: {claim}")
        supporting_evdience = kg.supported_by_map[claim]
        print(f"Evidence: {supporting_evdience}")
        refuting_evidence = kg.refuted_by_map[claim]
        print(f"Refuted evidence: {refuting_evidence}")
    else:
        print("No claim found in the graph.")
    
    ## TODO: respond to opponent


def test():
    store = InMemoryVectorStore()

    docs = [
        {
            'text': 'Jerry is 15 years old.',
            'metadata': {'uuid': '123', 'source': 'test_source'}
        },
        {
            'text': 'John is 170 cm tall.',
            'metadata': {'source': 'another_source', 'uuid': '456'}
        }
    ]

    for doc in docs:
        store.add(doc['text'], doc['metadata'])
    
    print(f"Length: {store.size()}")

    # Search for similar documents
    query = 'How tall is John?'
    results = store.search(query)
    print("Search results:")
    for result in results:
        print(f"ID: {result['id']}, Text: {result['text']}, Score: {result['score']}, Metadata: {result['metadata']}")


if __name__ == '__main__':
    # step1()
    step2()
    # test()
