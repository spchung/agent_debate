## atomic agents for specific tasks
import instructor
from typing import Literal
from typing import List
from pydantic import Field
from src.llm.client import llm
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
'''
    1. Claim summarizer agent

    input: 
    - topic
    - opponent's last message

    processing: 
    - identify claims from opponent's last message
    - summarize the claims
    - come up with questions that might verify the claims

    output:
    - questions for knowledge base
'''

class ClaimInqueryGeneratorInputSchema(BaseIOSchema):
    """ ClaimInqueryGeneratorInputSchema """
    topic: str = Field(None, title="The topic of the conversation")
    last_response: str = Field(None, title="The last response from the other agent")

class ClaimInqueryGeneratorOutputSchema(BaseIOSchema):
    """ ClaimInqueryGeneratorOutputSchema """
    questions: List[str] = Field(None, title="The questions for the knowledge base")

claim_inquery_generator_prompt = SystemPromptGenerator(
    background=[
        'You are a debate agent who has taken an opposing stance than your opponent.',
        'You are tasked with summarizing the claims made by your opponent in their last response.',
    ],
    steps=[
        'Carefully examine the last response from your opponent.',
        'Understnad the topic of the debate and the stance of your opponent.',
        'Identify the main claims made by your opponent.',
        'Come up with questions that might verify the claims made by your opponent.',
    ],
    output_instructions=[
        'Keep the questions concise and relevant to the claims made by your opponent.',
        'Make sure that the questions are relevant to the topic and the stance of your opponent.',
        'Output maximum 3 questions.',
    ]
)

claim_inquery_generator_agent = BaseAgent(
    BaseAgentConfig(
        client=instructor.from_openai(llm),
        model='gpt-4o-mini',
        temperature=0.7,
        system_prompt_generator=claim_inquery_generator_prompt,
        input_schema=ClaimInqueryGeneratorInputSchema,
        output_schema=ClaimInqueryGeneratorOutputSchema
    )
)

## 1. Invididual document summarization with opinionated stance
class TitleAndAuthorExtractorInputSchema(BaseIOSchema):
    """ TextSummarizerInputSchema """
    text: str = Field(None, title="The text to be processed")

class TitleAndAuthorExtractorOutputSchema(BaseIOSchema):
    """ TextSummarizerOutputSchema """
    author: str = Field(None, title="The author of the text")
    title: str = Field(None, title="The title of the text")

title_and_author_extractor_prompt = SystemPromptGenerator(
    background=[
        'Your task is to extract the title and author of the text.',
    ],
    steps=[
        'Read the beginning of the text carefully.',
    ],
    output_instructions=[
        'Make sure to extract the full name of the author.',
        'Make sure to extract the title of the text.',
    ]
)

title_and_author_extractor_agent = BaseAgent(
    BaseAgentConfig(
        client=instructor.from_openai(llm),
        model='gpt-4o-mini',
        temperature=0,
        system_prompt_generator=title_and_author_extractor_prompt,
        input_schema=TitleAndAuthorExtractorInputSchema,
        output_schema=TitleAndAuthorExtractorOutputSchema
    )
)