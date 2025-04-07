'''
1. resource claim generator agent
- input: resource, stance
- output: claims
'''

'''
2. evidence extraction agent 
- input: claim, resource
- output: evidence(s)
'''

'''
3. knowledge graph
- Nodes: claims, evidence
- Edges: 
    1. claim -supported_by-> evidence
    2. claim -refuted_by-> evidence
    3. claim -similar_to-> claim (weighted)
'''

import instructor
from typing import List
from pydantic import Field
from src.llm.client import llm
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

class ResourceClaimExtractionInputSchema(BaseIOSchema):
    """ ResourceClaimExtractionInputSchema """
    resource_raw_text: str = Field(None, title="The resource to be processed")
    topic: str = Field(None, title="The topic of the debate")
    stance: str = Field(None, title="The stance of the agent on the debate topic")

class ResourceClaimExtractionOutputSchema(BaseIOSchema):
    """ ResourceClaimExtractionOutputSchema """
    claims: List[str] = Field(None, title="The claims generated from the resource")
    resource_title: str = Field(None, title="The title of the resource")
    resource_author: str = Field(None, title="The author of the resource")

resource_claim_extraction_prompt = SystemPromptGenerator(
    background=[
        'You are a debate agent who has taken a stance on the topic of the debate.',
        'You are tasked with generating claims from the provided resource.',
        'The claims should be relevant to the topic and your stance on the topic.',
        'The claims should be concise and clear.'
    ],
    steps=[
        'Carefully examine the provided resource.',
        'Understand the topic of the debate and your stance on the topic.',
        'Follow this format when generating claims from the resource: [Subject/Policy/Action] [positive/negative evaluation] for [target/domain] because [primary reason] and [secondary reason if applicable].'
    ],
    output_instructions=[
        'Make sure that each claim is relevant to the topic and your stance on the topic.',
        'Output maximum 2 claims.'
    ]
)

claim_extraction_agent = BaseAgent(
    BaseAgentConfig(
        client=instructor.from_openai(llm),
        model='gpt-4o-mini',
        temperature=0.7,
        system_prompt_generator=resource_claim_extraction_prompt,
        input_schema=ResourceClaimExtractionInputSchema,
        output_schema=ResourceClaimExtractionOutputSchema
    )
)