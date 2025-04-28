import instructor
from pydantic import Field
from typing import List
from src.utils.logger import setup_logger
from src.llm.client import get_llm_instnace
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

logger = setup_logger()

class StatementAndRebuttalPairModel(BaseIOSchema):
    '''StatementAndRebuttalPairModel'''
    statement: str = Field(None, title='The statement made by the agent.')
    rebuttal: str = Field(None, title='The rebuttal made by the agent in response to the statement.')

class RebuttalQualityEvalInputSchema(BaseIOSchema):
    '''RebuttalQualityEvalInputSchema'''
    topic: str = Field(None, title='The topic of the debate.')
    statement_and_rebuttal_pairs: List[StatementAndRebuttalPairModel] = Field(None, title='All statement and rebuttal pairs made by the agent in the debate.')

class RebuttalQualityEvalOutputSchema(BaseIOSchema):
    '''RebuttalQualityEvalOutputSchema'''
    score: int = Field(None, title='The score for the rebuttal quality (1-10).')
    reasoning: str = Field(None, title='The reasoning for the score.')


rebuttal_quality_eval_prompt = SystemPromptGenerator(
    background=[
        'You are a debate judge. You will judge the rebuttal quality of a debate.',
        'You will have access to all statement and rebuttal pairs made by the agent in the debate.',
        'Consider each statement and rebuttal pair individually and then give an overall score.',
    ],
    steps = [
        'Consider the following metics when evaluating the rebuttal quality:',
        '1. How well does the rebuttal address the statement?',
        '2. How well does the rebuttal counter the arguments made in the statement?',
        '3. How strong is the rebuttal\'s reasoning and evidence?',
    ],
    output_instructions=[
        'Make sure to provide a score between 1 and 10, with 10 being the best.',
        'Provide a detailed reasoning for your score.',
    ]
)

rebuttal_quality_judge = BaseAgent(
    BaseAgentConfig(
        client = instructor.from_openai(get_llm_instnace()),
        model='gpt-4o-mini',
        temperature=0,
        system_prompt_generator=rebuttal_quality_eval_prompt,
        input_schema=RebuttalQualityEvalInputSchema,
        output_schema=RebuttalQualityEvalOutputSchema
    )
)