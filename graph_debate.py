from src.agents.graph.workers import (
    get_claim_extraction_agent, ResourceClaimExtractionInputSchema,
    get_evidence_extraction_agent, EvidenceExtractionInputSchema
)
import json
from src.debate.basic_history_manager import BasicHistoryManager
from src.utils.pdf_parser import PDFParser
from src.agents.graph.workers import DebateKnowledgeGraph, ClaimNode, EvidenceNode
from src.utils.in_mem_vector_store import InMemoryVectorStore
from src.utils.embedding import get_openai_embedding, cosine_similarity
from src.shared.models import AgnetConfig
from src.agents.graph.graph_agent_instructor import GraphDebateAgnet

file = '/Users/stephen/Nottingham/arbitrary_arbitration/knowledge_source/quantitative_easing/qe_turbulance.pdf'
TOPIC = "Quantitative Easing is a good policy for long-term economic growth."
STANCE = "for"

def step1():
    # Parse the PDF file
    pdf_parser = PDFParser()
    
    # Get the text from the PDF file
    raw_text = pdf_parser.pdf_to_text(file)

    # Extract claims from the text
    claim_res = get_claim_extraction_agent(num_of_claims=2).run(
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
    evidence_extraction_agent = get_evidence_extraction_agent(num_of_evidence=2)
    
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

    # print(kg.claim_similarity_map)
    # return 

    opponent_response = 'QE is an ineffective method of stimulating the economy. It leads to asset bubbles and income inequality.'

    ## find the closest claim in the graph
    claim = kg.get_most_relative_claim(opponent_response)

    print(f"Claim: {claim}")

    # find next best claim
    next_claim = kg.find_next_relative_claim(claim)
    print(f"Next claim: {next_claim}")

    # ## get related evidence
    # supporting_evdience = []
    # refuting_evidence = []

    # if claim:
    #     print(f"Claim: {claim}")
    #     supporting_evdience = kg.supported_by_map[claim]
    #     print(f"Evidence: {supporting_evdience}")
    #     refuting_evidence = kg.refuted_by_map[claim]
    #     print(f"Refuted evidence: {refuting_evidence}")
    # else:
    #     print("No claim found in the graph.")
    
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


from src.utils.logger import setup_logger
logger = setup_logger()

def main():
    shared_mem = BasicHistoryManager()
    moderator = AgnetConfig(id="moderator", name="Moderator", type="moderator")
    shared_mem.register_agent_moderator(moderator)

    TOPIC = "Quantitative Easing is a good policy for long-term economic growth."

    shared_mem.add_message(
        moderator, 
        f"""
            Today's topic is: '{TOPIC}'. 
        """)
    
    logger.info(f"Today's topic is: '{TOPIC}'")
    
    agent_1 = GraphDebateAgnet(
        topic=TOPIC,
        stance="against",
        agent_config=AgnetConfig(id="agent_1", name="Agent 1"),
        memory_manager=shared_mem,
        kb_path='knowledge_source/qe_mini',
        persist_kg_path='knowledge_graphs/agent_1_kg.json'
    )

    agent_2 = GraphDebateAgnet(
        topic=TOPIC,
        stance="for",
        agent_config=AgnetConfig(id="agent_2", name="Agent 2"),
        memory_manager=shared_mem,
        kb_path='knowledge_source/qe_mini',
        persist_kg_path='knowledge_graphs/agent_2_kg.json'
    )

    turns = lim = 5

    f = open("kg_debate_log.txt", "w")

    while turns > 0:
        print(f"====== Round {lim - (turns - 1)} start ======")
        res = agent_1.next_round_response(is_final = turns == 1)
        f.write(f"Agent 1: {res}\n")
        print(f"Agent 1: {res}")
        res = agent_2.next_round_response(is_final = turns == 1)
        f.write(f"Agent 2: {res}\n")
        print(f"Agent 2: {res}")
        print(f"====== Round {lim - (turns - 1)} End ======")
        turns -= 1

    f.close()


def test_cosinse_sim():
    a = get_openai_embedding('Jerry is 15 years old.')
    b = get_openai_embedding('John is 12 years old.')
    c = get_openai_embedding("The japanese yen has crashed against the dollar.")

    print(f"Cosine similarity: {cosine_similarity(a, b)}")
    print(f"Cosine similarity: {cosine_similarity(a, c)}")

if __name__ == '__main__':
    # step1()
    # step2()
    # test()
    # test_cosinse_sim()
    main()
