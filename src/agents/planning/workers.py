## atomic agents for specific tasks
import instructor
from typing import List, Literal
from pydantic import Field
from src.llm.client import llm
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

## 1. Invididual document summarization with opinionated stance
class TextSummarizerInputSchema(BaseIOSchema):
    """ TextSummarizerInputSchema """
    text: str = Field(None, title="The text to be processed")
    topic: str = Field(None, title="The topic of the debate")

class TextSummarizerOutputSchema(BaseIOSchema):
    """ TextSummarizerOutputSchema """
    debate_summary: str = Field(None, title="A summary of the text in the context of the debate topic and stance")
    key_points: List[str] = Field(None, title="Key points extracted from the text")
    text_stance: Literal['for', 'against'] = Field(None, title="The stance of the text on the debate topic")
    author: str = Field(None, title="The author of the text")
    title: str = Field(None, title="The title of the text")

class OpinionatedTextSummarizer:
    '''
    Text summarization agent with context of stance on debate and topic
    '''

    def __init__(self, topic: str, agent_stance: Literal['for', 'against']):
        self.topic = topic
        self.agent_stance = agent_stance
        self.agent = self.__build_agent()
    
    def __build_agent(self):
        text_summarizer_prompt = SystemPromptGenerator(
            background=[
                'You are a research agent.',
                'You are doing this to generate information that can be used in a debate.'
                'You have a specific stance on the topic of the debate.',
                f'The topic of the debate is "{self.topic}".'
                f'Your stance on the topic is "{self.agent_stance}".'
                'Construct a summary of the given text in the context of the debate topic.',
                'List out key points that will support the debate summary.',
                'Determine the stance that the text takes on the debate topic. It can either be for or against the topic.'
            ],
            steps=[
                'Read the provided text carefully.',
                'Understand the context of the debate and your stance on the topic.',
                'Construct a debate summary of the text that is relevant to the topic and your stance.',
                'List out key findings from the text that are mosr relative to the topic and your stance.',
                'Key points with data points should be given more weight.',
                'Determine the stance of the text on the debate topic.',
                'Extract the authors name and the title of the text.'
            ],
            output_instructions=[
                'Make sure the debate summary is relevant to the topic of the debate.',
                'Make sure that each key point is relevant to the topic of the debate and your stance.',
                'Make sure to include data points and evidence to support your key points if possible.',
                'Output maximum 5 key points.',
            ]
        )

        return BaseAgent(
            BaseAgentConfig(
                client=instructor.from_openai(llm),
                model='gpt-4o-mini',
                temperature=0.7,
                system_prompt_generator=text_summarizer_prompt,
                input_schema=TextSummarizerInputSchema,
                output_schema=TextSummarizerOutputSchema
            )
        )
    
    def set_topic(self, topic: str):
        self.topic = topic

    def run(self, input: TextSummarizerInputSchema):
        return self.agent.run(input)

