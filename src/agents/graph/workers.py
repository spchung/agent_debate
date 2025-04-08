'''
Components
1. resource claim generator agent
2. evidence extraction agent 
3. knowledge graph class
'''
import json
import instructor
from collections import defaultdict
from typing import List
from uuid import uuid4
from pydantic import Field
from src.llm.client import llm
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from src.utils.embedding import get_openai_embedding, cosine_similarity
from llama_index.core.schema import TextNode
from src.utils.in_mem_vector_store import InMemoryVectorStore
from src.utils.logger import setup_logger

logger = setup_logger()

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


def get_claim_extraction_agent(num_of_claims=3) -> List[str]:
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
            f'Output maximum {num_of_claims} claims.'
        ]
    )

    return BaseAgent(
        BaseAgentConfig(
            client=instructor.from_openai(llm),
            model='gpt-4o-mini',
            temperature=0.7,
            system_prompt_generator=resource_claim_extraction_prompt,
            input_schema=ResourceClaimExtractionInputSchema,
            output_schema=ResourceClaimExtractionOutputSchema
        )
    )

    # run the agent    

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

def get_evidence_extraction_agent(num_of_evidence=3):
    prompt = SystemPromptGenerator(
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
            f'Output maximum {num_of_evidence} evidences.'
        ]
    )

    return BaseAgent(
        BaseAgentConfig(
            client=instructor.from_openai(llm),
            model='gpt-4o-mini',
            temperature=0.7,
            system_prompt_generator=prompt,
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
    def __init__(self, claim, uuid=None):
        self.uuid = str(uuid4()) if uuid is None else uuid
        self.text = claim
        self.embedding = get_openai_embedding(claim)
        self.used = False
    
    def mark_used(self):
        self.used = True
        
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
    
    def to_llamaindex_node(self):
        return TextNode(text=self.text)

    def __dict___(self):
        return {
            'uuid': self.uuid,
            'text': self.text,
            # 'embedding': self.embedding
        }
    
    def __repr__(self):
        return f"ClaimNode(uuid={self.uuid}, text={self.text})"
        
class EvidenceNode():
    def __init__(self, text: str, is_support: bool, uuid=None):
        self.uuid = str(uuid4()) if uuid is None else uuid
        self.text = text
        self.is_support = is_support

    def __hash__(self):
        return hash(self.text)
    
    def model_dump(self):
        return {
            'uuid': self.uuid,
            'text': self.text,
            'is_support': self.is_support
        }

class DebateKnowledgeGraph:
    
    @classmethod
    def from_dict(cls, adict: dict):
        instance = cls()
        for claim in adict['claims']:
            instance.add_claim(claim['text'], uuid=claim['uuid'])
        
        for evidence in adict['evidence']:
            evidence_node = EvidenceNode(evidence['text'], evidence['uuid'])
            instance.evidence_nodes.add(evidence_node)
        
        for rel in adict['relations']:
            claim_node = next((c for c in instance.claim_nodes if c.uuid == rel[0]), None)
            evidence_node = next((e for e in instance.evidence_nodes if e.uuid == rel[2]), None)
            if claim_node and evidence_node:
                instance.add_pair(claim_node, evidence_node, rel[1] == "SUPPORTED_BY")
        
        # build lookup map
        for claim in instance.claim_nodes:
            instance.claim_node_lookup[claim.uuid] = claim

        return instance

    def __init__(self):
        self.claim_nodes = set()
        self.evidence_nodes = set()
        self.rels = set() # tuple (claim, is_supported (bool), evidence)
        
        ## maps
        self.supported_by_map = defaultdict(list) # claim to evidence
        self.refuted_by_map = defaultdict(list) # claim to evidence
        self.claim_similarity_map = defaultdict(list) # claim to claim (weighted)

        # claim node lookup
        self.vector_db = None
        self.claim_node_lookup = {} # uuid -> claim node
        self.prev_doc_length = 0 # number of documents in the vector store when last searched

    def __build_vector_db(self):
        
        store = InMemoryVectorStore()
        count = 0
        
        # add document to the vector store
        for claim in self.claim_nodes:
            store.add(claim.text, embedding=claim.embedding, metadata={'uuid': claim.uuid})
            count += 1
        
        self.vector_db = store
        self.prev_doc_length = count

    def get_most_relative_claim(self, query_str: str) -> ClaimNode | None:
        if not self.vector_db or self.prev_doc_length != len(self.claim_nodes):
            self.__build_vector_db()
        
        # find the claim that is closest to the input query string
        query_result = self.vector_db.search(query_str, limit=1)

        if len(query_result) < 1:
            logger.warning(f"No claim found for query: {query_str}")
            return None
        
        result = query_result[0]
        claim_uuid = result['metadata']['uuid']

        # find the claim node in the claim_nodes set
        result_claim =  self.claim_node_lookup.get(claim_uuid, None)
        if not result_claim:
            logger.warning(f"No claim found in the graph for uuid: {claim_uuid}")
        
        return result_claim
    
    def add_claim(self, claim: str, uuid:str=None) -> ClaimNode | None:
        claim_node = ClaimNode(claim, uuid=uuid)
        if claim_node in self.claim_nodes:
            logger.warning(f"Claim: {claim_node} already exists")
            return None
        self.claim_nodes.add(claim_node)
        self.claim_node_lookup[claim_node.uuid] = claim_node
        self.__build_corrolation(claim_node)
        return claim_node

    def find_next_relative_claim(self, claim: ClaimNode) -> ClaimNode | None:
        '''
        find the next most relative claim in the graph
        '''
        if claim not in self.claim_nodes:
            logger.warning(f"Claim: {claim} does not exist")
            return None
        
        # find the claim that is closest to the input query string
        claim_relations = self.claim_similarity_map[claim]
        if len(claim_relations) < 1:
            logger.warning(f"No relative claim found for claim: {claim}")
            return None
        
        # sort by similarity score
        claim_relations.sort(key=lambda x: x[1], reverse=True)
        # return the most similar claim
        most_similar_claim = claim_relations[0][0]
        if most_similar_claim == claim:
            logger.warning(f"No relative claim found for claim - default to using the same claim.")
            return claim
        return most_similar_claim

    def __build_corrolation(self, new_claim: ClaimNode) -> EvidenceNode | None:
        '''
        compare the new claim with all existing claims in the graph
        
        add similairity score to the corrolation map
        '''
        # TODO: implement similarity check
        if len(self.claim_nodes) < 1:
            return
        
        for claim in self.claim_nodes:
            if claim == new_claim:
                continue
            
            # calculate similarity
            similarity_score = cosine_similarity(new_claim.embedding, claim.embedding)
            self.claim_similarity_map[new_claim].append((claim, similarity_score))
            self.claim_similarity_map[claim].append((new_claim, similarity_score)) 

    def add_pair(self, claim_node: ClaimNode, evidence_node: EvidenceNode, is_support=True):
        if claim_node not in self.claim_nodes:
            logger.warning(f"Claim: {claim_node} does not exist")
            return
        
        rel = (claim_node, is_support, evidence_node)
        
        if rel in self.rels:
            logger.warning(f"Relation: {rel} already exists")
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
        # claim to evidence relations
        relations = [(claim.uuid, "SUPPORTED_BY" if is_support == True else "REFUTED_BY", evidence.uuid) for claim, is_support, evidence in self.rels]
        # claim to claim relations
        claim_relations = []

        for claim, related_claims in self.claim_similarity_map.items():
            for related_claim, similarity_score in related_claims:
                claim_relations.append((claim.uuid, float(similarity_score), related_claim.uuid))

        return {
            'claims': claims,
            'evidence': evidence,
            'relations': relations,
            'claim_similarity': claim_relations
        }