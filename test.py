from src.agents.base_agent_instructor import InstructorBaseAgent, DebateChatHistoryManager, AgnetConfig

shared_mem = DebateChatHistoryManager()

moderator = AgnetConfig(id="moderator", name="Moderator", type="moderator")
shared_mem.register_agent_moderator(moderator)

TOPIC = "Mobile phones should be banned in schools"

shared_mem.add_message(
    moderator, 
    f"""
        Today's topic is: '{TOPIC}'. 
        The againt side will argue for not banning mobile phones in schools, while the for side will argue for banning mobile phones in schools.
    """)


agent_1 = InstructorBaseAgent(
    topic=TOPIC,
    stance="against",
    agent_config=AgnetConfig(id="agent_1", name="Agent 1"),
    memory_manager=shared_mem
)

agent_2 = InstructorBaseAgent(
    topic=TOPIC,
    stance="for",
    agent_config=AgnetConfig(id="agent_2", name="Agent 2"),
    memory_manager=shared_mem
)

while True: 
    res = agent_1.next_round_response()
    print(f"Agent 1: {res}")
    res = agent_2.next_round_response()
    print(f"Agent 2: {res}")
    break


