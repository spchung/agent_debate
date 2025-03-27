from src.agents.kb.workers import ClaimInqueryGeneratorInputSchema, ClaimInqueryGeneratorOutputSchema, claim_inquery_generator_agent
from src.knowledge_base.pdf_kb import PdfKnowledgeBase
from src.debate.basic_history_manager import BasicHistoryManager
from src.shared.models import AgnetConfig
from src.agents.kb.kb_agent_instructor import KnowledgeBaseDebateAgent
from src.agents.planning.palnning_agent_instructor import PlanningDebateAgent


shared_mem = BasicHistoryManager()

moderator = AgnetConfig(id="moderator", name="Moderator", type="moderator")
shared_mem.register_agent_moderator(moderator)

TOPIC = "Quantitative Easing is a good policy for long-term economic growth."

shared_mem.add_message(
    moderator, 
    f"""
        Today's topic is: '{TOPIC}'. 
    """)

agent_1 = PlanningDebateAgent(
    topic=TOPIC,
    stance="against",
    agent_config=AgnetConfig(id="agent_1", name="Agent 1"),
    memory_manager=shared_mem,
    kb_path='knowledge_source/quantitative_easing'
)

agent_2 = PlanningDebateAgent(
    topic=TOPIC,
    stance="for",
    agent_config=AgnetConfig(id="agent_2", name="Agent 2"),
    memory_manager=shared_mem,
    kb_path='knowledge_source/quantitative_easing'
)

turns = lim = 5

f = open("planning_query_debate_log.txt", "w")

while turns > 0:
    print(f"====== Round {lim - (turns - 1)} start ======")
    res = agent_1.next_round_response(is_final=turns == 1)
    f.write(f"Agent 1: {res}\n")
    print(f"Agent 1: {res}")
    res = agent_2.next_round_response(is_final=turns == 1)
    f.write(f"Agent 2: {res}\n")
    print(f"Agent 2: {res}")
    print(f"====== Round {lim - (turns - 1)} End ======")
    turns -= 1

f.close()
    


