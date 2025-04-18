import os
import instructor
from pydantic import Field
from src.llm.client import llm
from src.utils.pdf_parser import PDFParser
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from src.agents.kb.workers import TitleAndAuthorExtractorInputSchema, title_and_author_extractor_agent

class DebateLogBeautifierInputSchema(BaseIOSchema):
    """ DebateLogBeautifierInputSchema """
    debate_log_raw_text: str = Field(None, title="The raw text of the debate log")

class DebateLogBeautifierOutputSchema(BaseIOSchema):
    """ DebateLogBeautifierOutputSchema """
    debate_log_md: str = Field(None, title="The markdown formatted debate log")


beautify_prompt = SystemPromptGenerator(
    background=[
        'You are a markdown beautifier agent.',
        'You are doing this to generate a markdown formatted debate log.'
    ],
    steps=[
        'Read the provided raw text carefully.',
        'Format the text into a markdown formatted debate log.'
    ],
    output_instructions=[
        'Make sure the topic is clearly stated at the beginning.',
        'Make sure the output is in markdown format.',
        'Do not alter the contenet.',
        'Start each debate turn with either [For_Agent] or [Against_Agent].',
        'Make sure the the transcript is in the correct order.',
        'Make sure to include proper headings, bullet points, and other markdown features as needed.'
    ]
)

beautifier_agent = BaseAgent(
    BaseAgentConfig(
        client=instructor.from_openai(llm),
        model='gpt-4o-mini',
        temperature=0.3,
        system_prompt_generator=beautify_prompt,
        input_schema=DebateLogBeautifierInputSchema,
        output_schema=DebateLogBeautifierOutputSchema,
    )
)


def list_available_resources(kb_path:str):
    parser  = PDFParser()
    resources = []
    for file in os.listdir(kb_path):
        if not file.endswith(".pdf"):
            continue

        raw_text = parser.pdf_to_text(f"{kb_path}/{file}")
        extraction_res = title_and_author_extractor_agent.run(
            TitleAndAuthorExtractorInputSchema(
                text=raw_text
            )
        )

        if extraction_res is None:
            continue

        resources.append({
            'title': extraction_res.title,
            'author': extraction_res.author,
        })
    return resources


