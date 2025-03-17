from atomic_agents.agents.base_agent import AgentMemory
from src.agents.base_agent import BaseDebateAgent, ResponseInputSchema

def main():
    
    shared_memory = AgentMemory()

    agent1 = BaseDebateAgent(shared_memory)
    agent2 = BaseDebateAgent(shared_memory)

    init_res = agent1.generate_response(
        ResponseInputSchema(
            topic="Why did the chicken cross the road?",
            last_response="",
            traits={
                "tone": "happy", 
                "attitude": "doubtful"
            }
        )
    )

    print(f"Agent 1: {init_res}")

    for _ in range(10):

        res1 = agent2.generate_response(
            ResponseInputSchema(
                topic="Why did the chicken cross the road?",
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
                topic="Why did the chicken cross the road?",
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