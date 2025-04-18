from src.knowledge_base.pdf_kb import PdfKnowledgeBase
from src.debate.basic_history_manager import BasicHistoryManager
from src.shared.models import AgnetConfig
from src.agents.kb.kb_agent_instructor import KnowledgeBaseDebateAgent
from src.agents.planning.palnning_agent_instructor import PlanningDebateAgent
from src.agents.basic.basic_agent_instructor import BasicDebateAgent
from src.agents.graph.graph_agent_instructor import GraphDebateAgnet
from src.debate.workers import beautifier_agent, list_available_resources

shared_mem = BasicHistoryManager()
moderator = AgnetConfig(id="moderator", name="Moderator", type="moderator")
shared_mem.register_agent_moderator(moderator)

topic = "Self-regulation by the AI industry is preferable to government regulation."
shared_mem.add_message(moderator, f"Today's topic is: '{topic}'. ")

# basic - for
basic_agent = BasicDebateAgent(
    topic=topic,
    stance="against",
    agent_config=AgnetConfig(id="basic", name="basic"),
    memory_manager=shared_mem
)

# agent 1 - planing
planning_agent = PlanningDebateAgent(
    topic=topic,
    stance="for",
    agent_config=AgnetConfig(id="planned", name="planned"),
    memory_manager=shared_mem,
    kb_path='knowledge_source/ai_regulation'
)

# kb - for 
# kb_agent = KnowledgeBaseDebateAgent(
#     topic=topic,
#     stance="for",
#     agent_config=AgnetConfig(id="kb_for", name="KB_For_AI_Regulation"),
#     memory_manager=shared_mem,
#     kb_path='knowledge_source/ai_regulation',
# )

# # graph
# graph_agent_config = AgnetConfig(id="graph_for", name="Graph_For_AI_Regulation")
# graph_agent = GraphDebateAgnet(
#     topic=topic,
#     stance="for",
#     agent_config=graph_agent_config,
#     memory_manager=shared_mem,
#     kb_path='knowledge_source/ai_regulation',
#     persist_kg_path=f'knowledge_graphs/{graph_agent_config.name}.json'
# )

# head to head debate

# one (for) vs three (against)
# can skop basic as for

# PLANNING vs all

iterations = [
    (planning_agent, [ basic_agent ]),
    # (planning_agent, [ basic_agent, kb_agent, graph_agent ]),
    # (kb_agent, [ basic_agent, planning_agent, graph_agent ]),
    # (graph_agent, [ basic_agent, kb_agent, planning_agent ]),
]

resources = list_available_resources('knowledge_source/ai_regulation')

def run_debates(for_agent, opponents:list, turns:int=5):
    for i in range(len(opponents)):
        # set turns
        turns = lim = turns

        opponent = opponents[i]

        # set stance
        for_agent.stance = "for"
        opponent.stance = "against"

        print(f"\n====== debat iteration: {i} ======\n")

        log_file_name = f"debate_results/{for_agent.agent_config.name}_vs_{opponent.agent_config.name}.txt"
        log = open(log_file_name, "w")

        shared_mem.reset()
        moderator = AgnetConfig(id="moderator", name="Moderator", type="moderator")
        shared_mem.register_agent_moderator(moderator)
        shared_mem.add_message(moderator, f"Today's topic is: '{topic}'. ")

        # register agents
        shared_mem.register_agent_debator(for_agent.agent_config)
        shared_mem.register_agent_debator(opponent.agent_config)

        log.write("Debate topic: " + topic + "\n")

        log.write("Available resources: \n")
        for i, res in enumerate(resources):
            log.write(f"{i+1} **{res['title']}** by {res['author']}\n")

        log.write("Debate Transcript: \n")

        # turns
        while turns > 0:
            log.write(f"====== Round {lim - (turns - 1)} ======\n\n")
            res = for_agent.next_round_response(is_final=turns == 1)
            log.write(f"for_agent: {res}\n\n")
            res = opponent.next_round_response(is_final=turns == 1)
            log.write(f"aganist_agent: {res}\n\n")
            turns -= 1
        log.close()

        raw_text = open(log_file_name, "r").read()

        md_log_file_name = f"debate_results/{for_agent.agent_config.name}_vs_{opponent.agent_config.name}.md"
        ff = open(md_log_file_name, "w+")

        res = beautifier_agent.run(
            beautifier_agent.input_schema(
                debate_log_raw_text=raw_text
            )
        )
        ff.write(res.debate_log_md)
        ff.close()


for for_agent, opponents in iterations:
    run_debates(for_agent, opponents)
    break