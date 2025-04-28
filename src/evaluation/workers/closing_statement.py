import instructor
from pydantic import Field
from typing import List
from src.utils.logger import setup_logger
from src.llm.client import get_llm_instnace
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

logger = setup_logger()

### Opening Statement Evaluation
class ClosingStatementEvalInputSchema(BaseIOSchema):
    '''ClosingStatementEvalInputSchema'''
    closing_statement: str = Field(None, title='The opening statement of the debate.')
    previous_statemnts: List[str] = Field(None, title='Previous statements made in the debate by this agent.')
    topic: str = Field(None, title='The topic of the debate.')

class ClosingStatementEvalOutputSchema(BaseIOSchema):
    '''ClosingStatementEvalOutputSchema'''
    score: int = Field(None, title='The score for the closing statement (1-10).')
    reasoning: str = Field(None, title='The reasoning for the score.')

closing_statement_eval_prompt = SystemPromptGenerator(
    background=[
        'You are a debate judge. You will judge the closing statement of a debate.',
        'You will have access to the all previous statements made by the agent, including the opening statement.',
    ],
    steps = [
        'Consider the following metics when evaluating the closing statement:',
        '1. How well does the agent summarize their position and key arguments?',
        '2. How well does the agent address the opposing arguments?',
        '3. How strong is the agent\'s conclusion and call to action?',

    ],
    output_instructions=[
        'Make sure to provide a score between 1 and 10, with 10 being the best.',
        'Provide a detailed reasoning for your score.',
    ]
)

closing_statement_judge = BaseAgent(
    BaseAgentConfig(
        client = instructor.from_openai(get_llm_instnace()),
        model='gpt-4o-mini',
        temperature=0.5,
        system_prompt_generator=closing_statement_eval_prompt,
        input_schema=ClosingStatementEvalInputSchema,
        output_schema=ClosingStatementEvalOutputSchema
    )
)