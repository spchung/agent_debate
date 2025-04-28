import instructor
from pydantic import Field
from typing import List
from src.utils.logger import setup_logger
from src.llm.client import get_llm_instnace
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

logger = setup_logger()


class CoherenceEvalInputSchema(BaseIOSchema):
    '''CoherenceEvalInputSchema'''
    topic: str = Field(None, title='The topic of the debate.')
    arguments: List[str] = Field(None, title='All arguments made by the agent in the debate.')

class CoherenceEvalOutputSchema(BaseIOSchema):
    '''CoherenceEvalOutputSchema'''
    score: int = Field(None, title='The score for the argument coherence (1-10).')
    reasoning: str = Field(None, title='The reasoning for the score.')

coherence_eval_prompt = SystemPromptGenerator(
    background=[
        'You are a debate judge tasked with evaluating the coherence of arguments in a debate.',
        'You will analyze all arguments presented by the agent during the debate.',
        'Assess each argument individually and provide an overall evaluation.',
    ],
    steps=[
        'Use the following criteria to evaluate argument coherence:',
        '1. Does each argument maintain a consistent position throughout the debate?',
        '2. Are there clear logical connections between premises and conclusions?',
        '3. Do subsequent arguments build logically on previous ones?',
    ],
    output_instructions=[
        'Assign a score between 1 and 10, where 10 represents the highest coherence.',
        'Provide a detailed explanation justifying your score.',
    ]
)

coherence_quality_judge = BaseAgent(
    BaseAgentConfig(
        client = instructor.from_openai(get_llm_instnace()),
        model='gpt-4o-mini',
        temperature=0,
        system_prompt_generator=coherence_eval_prompt,
        input_schema=CoherenceEvalInputSchema,
        output_schema=CoherenceEvalOutputSchema
    )
)