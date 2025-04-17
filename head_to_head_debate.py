from src.knowledge_base.pdf_kb import PdfKnowledgeBase
from src.debate.basic_history_manager import BasicHistoryManager
from src.shared.models import AgnetConfig
from src.agents.kb.kb_agent_instructor import KnowledgeBaseDebateAgent
from src.agents.planning.palnning_agent_instructor import PlanningDebateAgent
from src.agents.basic.basic_agent_instructor import BasicDebateAgent

shared_mem = BasicHistoryManager()
moderator = AgnetConfig(id="moderator", name="Moderator", type="moderator")
shared_mem.register_agent_moderator(moderator)

topic = "Self-regulation by the AI industry is preferable to government regulation."

shared_mem.add_message(moderator, f"Today's topic is: '{topic}'. ")

# agent 1 - planing
agent_1 = PlanningDebateAgent(
    topic=topic,
    stance="against",
    agent_config=AgnetConfig(id="palnned_agent_against", name="Planned_Against_AI_Regulation"),
    memory_manager=shared_mem,
    kb_path='knowledge_source/ai_regulation'
)

# basic - for
# agent_2 = BasicDebateAgent(
#     topic=topic,
#     stance="for",
#     agent_config=AgnetConfig(id="basic_for", name="Basic_For_AI_Regulation"),
#     memory_manager=shared_mem
# )

# kb - for 
agent_2 = KnowledgeBaseDebateAgent(
    topic=topic,
    stance="for",
    agent_config=AgnetConfig(id="kb_for", name="KB_For_AI_Regulation"),
    memory_manager=shared_mem,
    kb_path='knowledge_source/ai_regulation',
)

turns = lim = 3

f = open("debate_logs/planning_vs_kb_debate_log.txt", "w")

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