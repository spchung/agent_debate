import instructor
from typing import Any, Literal
from pydantic import Field
from src.llm.client import llm
from abc import ABC, abstractmethod
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema, AgentMemory
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

## atomic agent 
class ResponseInputSchema(BaseIOSchema):
    """ ResponseInputSchema """
    topic: str = Field(None, title="The topic of the conversation")
    last_response: str = Field(None, title="The last response from the other agent")
    context: Any | None = Field(None, title="The context of the conversation")

class ResponseOutputSchema(BaseIOSchema):
    """ ResponseOutputSchema """
    message: str = Field(None, title="The response of the agent")

class BaseDebateAgent(ABC):
    @abstractmethod
    def build_response_prompt(self) -> SystemPromptGenerator:
        pass
    @abstractmethod
    def build_response_agent(self) -> BaseAgent:
        pass
    @abstractmethod
    def generate_response(self, input: ResponseInputSchema) -> ResponseOutputSchema:
        pass

class GenericDebateAgent(BaseDebateAgent):
    def __init__(self, position: Literal["FOR", "AGAINST"], memory: AgentMemory=None):
        self.memory = memory # shared memory between agents
        self.position = position
        self.response_agent = self.build_response_agent()
    
    def build_response_prompt(self) -> SystemPromptGenerator:
        return SystemPromptGenerator(
            background=[
                'You are a debate agent that take a position on the presented topic.',
                f'You are arguing {self.position.lower()} the topic.',
            ],
            steps=[
                'Analyze the topic and the previous response from your opponent.',
                'Use the information provided to generate a response.',
                "Identify 2-3 specific claims from your opponent's last response and address them directly",  
            ],
            output_instructions=[
                'No need to repeat the topic or the last response.',
                'Keep the response concise to four or five sentences.',
                f'Make sure that your response is relevant and arguing {self.position.lower()} the topic.',
            ]
        )

    def build_response_agent(self) -> BaseAgent:
        prompt = self.build_response_prompt()
        return BaseAgent(
            BaseAgentConfig(
                client=instructor.from_openai(
                    llm
                ),
                memory=self.memory,
                model='gpt-4o-mini',
                system_prompt_generator=prompt,
                input_schema=ResponseInputSchema,
                output_schema=ResponseOutputSchema
            )
        )
        
    def generate_response(self, input: ResponseInputSchema) -> ResponseOutputSchema:
        """ Generate a response based on the input """
        res = self.response_agent.run(input)
        return res.message

class ReasonDebateAgent(BaseDebateAgent):
    """ A debate agent that takes in the topic first then prepares a knowledge base for later user """
    pass