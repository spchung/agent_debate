import instructor
from pydantic import Field
from src.llm.client import llm
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema, AgentMemory
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

# class Topic:
#     """ Topic """
#     name: str = Field(None, title="The topic of the conversation")
    
class TwinMemory:
    """
        A shared memory between two agents
        update both agents' memory with different perspectives
    """
    def __init__(self, memory1: AgentMemory, memory2: AgentMemory):
        self.memory1 = memory1
        self.memory2 = memory2


## atomic agent 
class ResponseInputSchema(BaseIOSchema):
    """ ResponseInputSchema """
    topic: str = Field(None, title="The topic of the conversation")
    last_response: str = Field(None, title="The last response from the other agent")
    traits: dict = Field(None, title="The traits of the agent")

class ResponseOutputSchema(BaseIOSchema):
    """ ResponseOutputSchema """
    message: str = Field(None, title="The response of the agent")

response_prompt = SystemPromptGenerator(
    background=[
        'You are a research scientist discussing a topic with another research scientist.',
    ],
    steps=[
        'Use the information provided to generate a response.',
    ],
    output_instructions=[
        'No need to repeat the topic or the last response.',
        'Be professional and respectful.',
    ]
)

class BaseDebateAgent:
    def __init__(self, memory: AgentMemory):
        self.memory = memory # shared memory between agents
        self.response_agent = self.__build_response_agent()

    def __build_response_agent(self):
        return BaseAgent(
            BaseAgentConfig(
                client=instructor.from_openai(
                    llm
                ),
                model='gpt-4o-mini',
                system_prompt_generator=response_prompt,
                input_schema=ResponseInputSchema,
                output_schema=ResponseOutputSchema
            )
        )
        

    def generate_response(self, input: ResponseInputSchema) -> ResponseOutputSchema:
        """
            Generate a response based on the input
        """
        res = self.response_agent.run(input)
        return res.message
