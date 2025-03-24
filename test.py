from src.agents.basic.basic_agent_instructor import BasicDebateAgent
from src.debate.basic_history_manager import BasicHistoryManager
from src.shared.models import AgnetConfig

shared_mem = BasicHistoryManager()

moderator = AgnetConfig(id="moderator", name="Moderator", type="moderator")
shared_mem.register_agent_moderator(moderator)

TOPIC = "Mobile phones should be banned in schools"

shared_mem.add_message(
    moderator, 
    f"""
        Today's topic is: '{TOPIC}'. 
        The againt side will argue for not banning mobile phones in schools, while the for side will argue for banning mobile phones in schools.
    """)


agent_1 = BasicDebateAgent(
    topic=TOPIC,
    stance="against",
    agent_config=AgnetConfig(id="agent_1", name="Agent 1"),
    memory_manager=shared_mem
)

agent_2 = BasicDebateAgent(
    topic=TOPIC,
    stance="for",
    agent_config=AgnetConfig(id="agent_2", name="Agent 2"),
    memory_manager=shared_mem
)

turns = lim = 10

f = open("log.txt", "w")

while turns > 0:
    print(f"====== Round {lim - (turns - 1)} start ======")
    res = agent_1.next_round_response()
    f.write(f"Agent 1: {res}\n")
    print(f"Agent 1: {res}")
    res = agent_2.next_round_response()
    f.write(f"Agent 2: {res}\n")
    print(f"Agent 2: {res}")
    print(f"====== Round {lim - (turns - 1)} End ======")
    turns -= 1

f.close()
    


