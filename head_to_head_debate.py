from typing import Literal
from src.knowledge_base.pdf_kb import PdfKnowledgeBase
from src.debate.basic_history_manager import BasicHistoryManager
from src.shared.models import AgnetConfig
from src.agents.kb.kb_agent_instructor import KnowledgeBaseDebateAgent
from src.agents.planning.planning_agent_instructor import PlanningDebateAgent
from src.agents.basic.basic_agent_instructor import BasicDebateAgent
from src.agents.graph.graph_agent_instructor import GraphDebateAgnet
from src.debate.workers import beautifier_agent, list_available_resources

def debate_agent_factory(
    agent_config: AgnetConfig,
    topic: str,
    stance: str,
    kb_path: str = None,
    persist_kg_path: str = None,
    agent_type: Literal["basic", "planning", "kb", "graph"] = "basic" 
):
    if agent_type not in ["basic", "planning", "kb", "graph"]:
        raise ValueError(f"Invalid agent type: {agent_type}. Must be one of ['basic', 'planning', 'kb', 'graph']")
    
    if agent_type == "basic":
        return BasicDebateAgent(topic, stance, agent_config)
    elif agent_type == "planning":
        return PlanningDebateAgent(topic, stance, agent_config, kb_path)
    elif agent_type == "kb":
        return KnowledgeBaseDebateAgent(topic, stance, agent_config, kb_path)
    elif agent_type == "graph":
        return GraphDebateAgnet(topic,
            stance,
            agent_config=agent_config,
            kb_path=kb_path,
            persist_kg_path=persist_kg_path
        )
    
def generate_closing(shared_mem: BasicHistoryManager, agent, moderator):
    # Create temporary separate memory managers for final statements
    temp_mem = BasicHistoryManager()
    
    # Copy all messages to the temporary managers
    for msg in shared_mem.messages:
        temp_mem.add_message(msg.agent_config, msg.message)
    
    # Register agents in the temporary memory managers
    temp_mem.register_agent_moderator(moderator)
    temp_mem.register_agent_debator(agent.agent_config)
    
    # Store original memory managers
    orig_mem = agent.memory_manager
    
    # Set temporary memory managers
    agent.memory_manager = temp_mem
    
    # Generate final statements
    res = agent.next_round_response(is_final=True)
    
    # Restore original memory managers
    agent.memory_manager = orig_mem
    
    # Add final statements to the original memory manager
    shared_mem.add_message(agent.agent_config, res)

    return res

def run_debate(for_agent, opponent_agent, resources: list, turns:int=5):
    total_turns = turns
    shared_mem = BasicHistoryManager()
    moderator = AgnetConfig(id="moderator", name="Moderator", type="moderator")
    shared_mem.register_agent_moderator(moderator)
    
    # set up topic
    topic = "Self-regulation by the AI industry is preferable to government regulation."
    shared_mem.add_message(moderator, f"Today's topic is: '{topic}'. ")

    # grant memory access to agents
    for_agent.memory_manager = shared_mem
    opponent_agent.memory_manager = shared_mem

    # register agents in shared memory
    shared_mem.register_agent_debator(for_agent.agent_config)
    shared_mem.register_agent_debator(opponent_agent.agent_config)
    
    # set turns
    lim = turns

    # set up log file
    log_file_name = f"debate_results/{for_agent.agent_config.name}_vs_{opponent_agent.agent_config.name}.txt"
    log = open(log_file_name, "w")

    # write topic and headers
    log.write("Debate topic: " + topic + "\n")
    log.write("Available resources: \n")

    # write available resources
    for i, res in enumerate(resources):
        log.write(f"{i+1} **{res['title']}** by {res['author']}\n")

    log.write("Debate Transcript: \n")

    # turns
    while turns > 0:
        
        # Closing 
        if turns == 1:
            log.write(f"====== Closing Statements ======\n\n")
            for_res = generate_closing(shared_mem, for_agent, moderator)
            log.write(f"[for_agent]: {for_res}\n\n")
            opponent_res = generate_closing(shared_mem, opponent_agent, moderator)
            log.write(f"[aganist_agent]: {opponent_res}\n\n")
        # Opening
        elif turns == total_turns:
            log.write(f"====== Opening Statements ======\n\n")
            res = for_agent.next_round_response(is_opening=True)
            log.write(f"[for_agent]: {res}\n\n")
            res = opponent_agent.next_round_response(is_opening=True)
            log.write(f"[aganist_agent]: {res}\n\n")
        # Regular rounds
        else:
            log.write(f"====== Round {lim - (turns - 1)} ======\n\n")
            res = for_agent.next_round_response()
            log.write(f"[for_agent]: {res}\n\n")
            res = opponent_agent.next_round_response()
            log.write(f"[aganist_agent]: {res}\n\n")
        turns -= 1

    log.close()

    # gen markdown
    raw_text = open(log_file_name, "r").read()
    md_log_file_name = f"debate_results/{for_agent.agent_config.name}_vs_{opponent_agent.agent_config.name}.md"
    ff = open(md_log_file_name, "w+")

    res = beautifier_agent.run(
        beautifier_agent.input_schema(
            debate_log_raw_text=raw_text
        )
    )
    ff.write(res.debate_log_md)
    ff.close()

def main(debate_turns=5):
    # top lvl vars:
    TOPIC = "Self-regulation by the AI industry is preferable to government regulation."
    RESOURCE_DIR = "knowledge_source/ai_regulation"

    # resource list
    resources = list_available_resources(RESOURCE_DIR)

    iterations = [
        # ('planning', [ 'basic', 'kb', 'graph' ]),
        # ('kb', [ 'basic', 'planning', 'graph' ]),
        ('graph', [ 'basic', 'planning', 'kb' ]),
    ]
    
    for for_agent_type, opponents_type in iterations:
        for_agent = debate_agent_factory(
            agent_config=AgnetConfig(id=f"{for_agent_type}_for", name=f"{for_agent_type}_for", type='debator'),
            topic=TOPIC,
            stance="for",
            kb_path='knowledge_source/ai_regulation',
            agent_type=for_agent_type
        )

        for i, opponent_type in enumerate(opponents_type):
            opponent = debate_agent_factory(
                agent_config=AgnetConfig(id=f"{opponent_type}_against_{i}", name=f"{opponent_type}_against_{i}", type='debator'),
                topic=TOPIC,
                stance="against",
                kb_path='knowledge_source/ai_regulation',
                agent_type=opponent_type,
            )

            run_debate(for_agent, opponent, resources=resources, turns=debate_turns)

if __name__ == "__main__":
    main(debate_turns=6)