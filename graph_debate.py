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
        # kb_path='knowledge_source/quantitative_easing',
        kb_path='knowledge_source/qe_mini',
        persist_kg_path='knowledge_graphs/agent_1_kg.json'
    )

    agent_2 = GraphDebateAgnet(
        topic=TOPIC,
        stance="for",
        agent_config=AgnetConfig(id="agent_2", name="Agent 2"),
        memory_manager=shared_mem,
        # kb_path='knowledge_source/quantitative_easing',
        kb_path='knowledge_source/qe_mini',
        persist_kg_path='knowledge_graphs/agent_2_kg.json'
    )

    turns = lim = 7

    f = open("kg_debate_log.txt", "w")

    while turns > 0:
        print(f"====== Round {lim - (turns - 1)} start ======")
        res = agent_1.next_round_response(is_final = turns == 1)
        f.write(f"Agent 1: {res}\n\n")
        print(f"Agent 1: {res}")
        res = agent_2.next_round_response(is_final = turns == 1)
        f.write(f"Agent 2: {res}\n\n")
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
