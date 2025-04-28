import instructor
from pydantic import Field
from typing import List
from src.utils.logger import setup_logger
from src.llm.client import get_llm_instnace
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

logger = setup_logger()


class ArgumentQualityEvalInputSchema(BaseIOSchema):
    '''ArgumentQualityEvalInputSchema'''
    topic: str = Field(None, title='The topic of the debate.')
    arguments: List[str] = Field(None, title='All arguments made by the agent in the debate.')

class ArgumentQualityEvalOutputSchema(BaseIOSchema):
    '''ArgumentQualityEvalOutputSchema'''
    score: int = Field(None, title='The score for the argument quality (1-10).')
    reasoning: str = Field(None, title='The reasoning for the score.')

argument_quality_eval_prompt = SystemPromptGenerator(
    background=[
        'You are a debate judge. You will judge the argument quality of a debate.',
        'You will have access to all arguments made by the agent in the debate.',
        'Consider each argument individually and then give an overall score.',
    ],
    steps = [
        'Consider the following metrics when evaluating the argument quality:',
        '1. How well does each argument support the agent\'s position?',
        '2. How well does each argument address counterarguments?',
        '3. How strong is the reasoning and evidence provided for each argument?',  
    ],
    output_instructions=[
        'Make sure to provide a score between 1 and 10, with 10 being the best.',
        'Provide a detailed reasoning for your score.',
    ]
)

argument_quality_judge = BaseAgent(
    BaseAgentConfig(
        client = instructor.from_openai(get_llm_instnace()),
        model='gpt-4o-mini',
        temperature=0,
        system_prompt_generator=argument_quality_eval_prompt,
        input_schema=ArgumentQualityEvalInputSchema,
        output_schema=ArgumentQualityEvalOutputSchema
    )
)