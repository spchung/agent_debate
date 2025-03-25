from atomic_agents.agents.base_agent import AgentMemory
from src.agents.base_agent_atomic import GenericDebateAgent, ResponseInputSchema

def main():
    shared_memory = AgentMemory()
    agent1 = GenericDebateAgent(
        position="FOR",
        memory=shared_memory
    )
    
    agent2 = GenericDebateAgent(
        position="AGAINST",
        memory=shared_memory
    )

    topic = "Video game violence leads to real-world violence"

    init_res = agent1.generate_response(
        ResponseInputSchema(
            topic=topic,
            last_response="",
            traits={
                "tone": "happy", 
                "attitude": "doubtful"
            }
        )
    )

    print(f"Agent 1: {init_res}")

    for _ in range(5):

        res1 = agent2.generate_response(
            ResponseInputSchema(
                topic=topic,
                # context = "You are a old mean lady who is sick of debates. All you want to do is to bake cookies",
                last_response=init_res,
                traits={
                    "tone": "happy", 
                    "attitude": "doubtful"
                }
            )
        )
        
        print(f"Agent 2: {res1}")

        res2 = agent1.generate_response(
            ResponseInputSchema(
                topic=topic,
                last_response=res1,
                traits={
                    "tone": "happy", 
                    "attitude": "doubtful"
                }
            )
        )

        print(f"Agent 1: {res2}")

        print()

    print("Hello, world!")
    print("Goodbye, world!")


if __name__ == "__main__":
    main()