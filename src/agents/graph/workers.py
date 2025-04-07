'''
Components
1. resource claim generator agent
2. evidence extraction agent 
3. knowledge graph class
'''

import instructor
from collections import defaultdict
from pydantic import BaseModel
from typing import List
from uuid import uuid4
from pydantic import Field
from src.llm.client import llm
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from src.utils.embedding import get_openai_embedding

'''
1. resource claim generator agent
- input: resource, stance
- output: claims
'''
class ResourceClaimExtractionInputSchema(BaseIOSchema):
    """ ResourceClaimExtractionInputSchema """
    resource_raw_text: str = Field(None, title="The resource to be processed")
    topic: str = Field(None, title="The topic of the debate")
    stance: str = Field(None, title="The stance of the agent on the debate topic")

class ResourceClaimExtractionOutputSchema(BaseIOSchema):
    """ ResourceClaimExtractionOutputSchema """
    claims: List[str] = Field(None, title="The claims generated from the resource")

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

'''
2. evidence extraction agent 
- input: claim, resource
- output: evidence(s)
'''
class EvidenceExtractionInputSchema(BaseIOSchema):
    """ EvidenceExtractionInputSchema """
    claim: str = Field(None, title="The claim to be supported or refuted")
    resource_raw_text: str = Field(None, title="The resource to be processed")
    topic: str = Field(None, title="The topic of the debate")
    stance: str = Field(None, title="The stance of the agent on the debate topic")

class EvidenceExtractionOutputSchema(BaseIOSchema):
    """ EvidenceExtractionOutputSchema """
    evidence: List[str] = Field(None, title="The evidence generated from the resource")

evidence_extraction_prompt = SystemPromptGenerator(
    background=[
        'You are a debate agent who has taken a stance on the topic of the debate.',
        'You are tasked with generating evidence to support or refute the provided claim.',
        'The evidence should be relevant to the topic and your stance on the topic.',
        'The evidence should be concise and clear.'
    ],
    steps=[
        'Carefully examine the provided resource.',
        'Understand the topic of the debate and your stance on the topic.',
        'Prioritize evidence information with data points and statistics.',
        'Do not include any personal opinions or beliefs.',
    ],
    output_instructions=[
        'Make sure that each evidence is relevant to claim',
        'Output maximum 2 evidences.'
    ]
)

evidence_extraction_agent = BaseAgent(
    BaseAgentConfig(
        client=instructor.from_openai(llm),
        model='gpt-4o-mini',
        temperature=0.7,
        system_prompt_generator=evidence_extraction_prompt,
        input_schema=EvidenceExtractionInputSchema,
        output_schema=EvidenceExtractionOutputSchema
    )
)

'''
3. knowledge graph class
- Nodes: claims, evidence
- Edges: 
    1. claim -supported_by-> evidence
    2. claim -refuted_by-> evidence
    3. claim -similar_to-> claim (weighted)
'''
class ClaimNode():
    def __init__(self, claim):
        self.uuid = str(uuid4())
        self.text = claim
        self.embedding = get_openai_embedding(claim)
    
    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other):
        # Two nodes are equal if they have the same text
        if not isinstance(other, ClaimNode):
            return False
        return self.text == other.text

    def model_dump(self):
        return {
            'uuid': self.uuid,
            'text': self.text,
        }
        
class EvidenceNode():
    def __init__(self, text: str):
        self.uuid = str(uuid4())
        self.text = text

    def __hash__(self):
        return hash(self.text)
    
    def model_dump(self):
        return {
            'uuid': self.uuid,
            'text': self.text
        }

class DebateKnowledgeGraph:
    def __init__(self):
        self.claim_nodes = set()
        self.evidence_nodes = set()
        self.rels = set() # tuple (claim, is_supported (bool), evidence)
        
        ## maps
        self.supported_by_map = defaultdict(list) # claim to evidence
        self.refuted_by_map = defaultdict(list) # claim to evidence
        self.corrolation_map = defaultdict(list) # claim to claim (weighted)
    
    def add_claim(self, claim: str) -> ClaimNode | None:
        claim_node = ClaimNode(claim)
        if claim_node in self.claim_nodes:
            print(f"[Warning] Claim: {claim_node} already exists")
            return None
        self.claim_nodes.add(claim_node)
        return claim_node

    def add_pair(self, claim_node: ClaimNode, evidence: str, is_support=True):
        if claim_node not in self.claim_nodes:
            print(f"[Warning] Claim: {claim_node} does not exist")
            return
        
        evidence_node = EvidenceNode(evidence)
        rel = (claim_node, is_support, evidence_node)
        
        if rel in self.rels:
            print(f"[Warning] Relation: {rel} already exists")
            return
        
        # update maps
        if is_support:
            self.supported_by_map[claim_node].append(evidence_node)
        else:
            self.refuted_by_map[claim_node].append(evidence_node)
        
        # add to sets
        self.rels.add(rel)
        self.evidence_nodes.add(evidence_node)

    def to_json(self):
        claims = [claim.model_dump() for claim in self.claim_nodes]
        evidence = [evidence.model_dump() for evidence in self.evidence_nodes]
        relations = [(claim.uuid, "SUPPORTED_BY" if is_support == True else "REFUTED_BY", evidence.uuid) for claim, is_support, evidence in self.rels]
        
        return {
            'claims': claims,
            'evidence': evidence,
            'relations': relations
        }