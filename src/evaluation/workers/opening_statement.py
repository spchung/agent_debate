import instructor
from pydantic import Field
from src.utils.logger import setup_logger
from src.llm.client import get_llm_instnace
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

logger = setup_logger()

### Opening Statement Evaluation
class OpeningStatementEvalInputSchema(BaseIOSchema):
    '''OpeningStatementEvalInputSchema'''
    opening_statement: str = Field(None, title='The opening statement of the debate.')
    topic: str = Field(None, title='The topic of the debate.')

class OpeningStatementEvalOutputSchema(BaseIOSchema):
    '''OpeningStatementEvalOutputSchema'''
    score: int = Field(None, title='The score for the opening statement (1-10).')
    reasoning: str = Field(None, title='The reasoning for the score.')

opening_statement_eval_prompt = SystemPromptGenerator(
    background=[
        'You are a debate judge. You will judge the opening statement of a debate.',
    ],
    steps = [
        'Consider the following metics when evaluating the opening statement:',
        '1. How well does the agent frame their position and introduce key arguments?'
    ],
    output_instructions=[
        'Make sure to provide a score between 1 and 10, with 10 being the best.',
        'Provide a detailed reasoning for your score.',
    ]
)

opening_statement_judge = BaseAgent(
    BaseAgentConfig(
        client = instructor.from_openai(get_llm_instnace()),
        model='gpt-4o-mini',
        temperature=0.5,
        system_prompt_generator=opening_statement_eval_prompt,
        input_schema=OpeningStatementEvalInputSchema,
        output_schema=OpeningStatementEvalOutputSchema
    )
)