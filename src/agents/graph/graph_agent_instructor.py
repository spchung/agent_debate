'''
This agent consumes knowledge base similar to the planning agent.
But instead of making stance summary for each resource, information from all sources are pooled into a knowlege graph

How to build the graph
- iterate over each resource
  1. first read
    - generate claims to support your stance (2 - 3 claims)
    - if claim is similar to claims already generated in previous resouces -> use the same claim 
    - SIMILARITY CHECK (embedding or llm)
  2. second read (for each claim)
    - find evidence that supports or refutes the claim
    - add edge between claim and evidence

Inference:
- extract keypoints from the opponent's response
- find closest claim in the graph 
- find evidence that supports or refutes the claim
'''

import instructor
import json, os
from typing import List, Tuple
from typing import Literal
from src.llm.client import llm
from src.debate.basic_history_manager import BasicHistoryManager
from src.shared.models import AgnetConfig, ResponseModel
from src.utils.pdf_parser import PDFParser
from src.agents.graph.workers import (
    get_claim_extraction_agent, ResourceClaimExtractionInputSchema,
    get_evidence_extraction_agent, EvidenceExtractionInputSchema
)
from src.agents.graph.workers import DebateKnowledgeGraph, ClaimNode, EvidenceNode
from src.utils.text_split import split_text_by_sentences_and_length
from src.utils.logger import setup_logger

logger = setup_logger()
client = instructor.from_openai(llm)
class GraphDebateAgnet:
    def __init__(self,
        topic: str,
        stance: Literal["for", "against"],
        agent_config: AgnetConfig,
        memory_manager: BasicHistoryManager,
        kb_path: str,
        persist_kg_path: str = None, # if not provided, each run will require rebuilding a new graph
    ):
        self.memory_manager = memory_manager
        self.topic = topic
        self.stance = stance
        self.agent_config = agent_config

        # resigter agent
        self.memory_manager.register_agent_debator(agent_config)

        # knowledge graph
        self.pdf_parser = PDFParser()
        self.kg = self.__build_kg(kb_path, persist_kg_path)
    
    def __build_kg(self, kb_path, persist_kg_path):
        kg = DebateKnowledgeGraph()
        for file in os.listdir(kb_path):
            if not file.endswith('.pdf'):
                continue

            self.__process_document(
                f"{kb_path}/{file}",
                kg
            )
        
        # save kg in json form if persist_kg_path is provided
        if persist_kg_path:
            with open(persist_kg_path, 'w') as f:
                json.dump(kg.to_json(), f, indent=4)

        return kg

    def __process_document(
            self, 
            file,
            kg: DebateKnowledgeGraph
        ):
        doc_raw_text = self.pdf_parser.pdf_to_text(file)
        
        # chunking
        chunks = [doc_raw_text]
        if len(doc_raw_text) > 10000:
            chunks = split_text_by_sentences_and_length(doc_raw_text, 10000)
        
        logger.info(f"Document {file} has been chunked into {len(chunks)} chunks")

        num_of_claims = 2
        num_of_evidence = 3
        if len(chunks) > 1:
            num_of_claims = 1
            num_of_evidence = 2

        claim_extraction_agent = get_claim_extraction_agent(num_of_claims=num_of_claims)
        evidence_extraction_agent = get_evidence_extraction_agent(num_of_evidence=num_of_evidence)

        for raw_text_chunk in chunks:
            claim_res = claim_extraction_agent.run(
                ResourceClaimExtractionInputSchema(
                    resource_raw_tex=raw_text_chunk,
                    topic=self.topic,
                    stance=self.stance
                )
            )

            # process claims from each chunk
            for claim in claim_res.claims:
                logger.info(f"PROCESSING Claim: {claim}")
                    
                claim_node = kg.add_claim(claim)
                
                evidence_res = evidence_extraction_agent.run(
                    EvidenceExtractionInputSchema(
                        claim=claim,
                        resource_raw_text=raw_text_chunk,
                        stance=self.stance
                    )
                )

                for evidence in evidence_res.evidence:
                    kg.add_pair(
                        claim_node, 
                        EvidenceNode(evidence, is_support=True),
                        is_support=True
                    )
                
                counter_evidence_res = evidence_extraction_agent.run(
                    EvidenceExtractionInputSchema(
                        claim=claim,
                        resource_raw_text=raw_text_chunk,
                        stance='against' if self.stance == 'for' else 'for'
                    )
                )

                for evidence in counter_evidence_res.evidence:
                    kg.add_pair(
                        claim_node, 
                        EvidenceNode(evidence, is_support=False),
                        is_support=False
                    )
    
    def __select_claim_from_kg(self, opponent_response) -> Tuple[ClaimNode, List[EvidenceNode], List[EvidenceNode]]:
        '''
        steps:
        1. extract keypoints from the opponent's response (TODO)
        2. find closest claim in the graph by similarity by way of embedding
        3. check if claim is already used in previous rounds
            3.1. if not used, find evidence that supports or refutes the claim
            3.2. if used, find next most relative claim (using the claim-RELATED_TO->claim relationship)
        4. get evidence and counter evidence for the claim
        5. return claim, evidence, counter_evidence
        '''

        # 2. get most relative claim to the opponent's response
        claim = self.kg.get_most_relative_claim(opponent_response)
        if not claim:
            return None, None, None
        
        # 3. claim already used in previous rounds
        if claim.used:
            # find next most relative claim
            claim = self.kg.find_next_relative_claim(claim)

            if not claim:
                return None, None, None

        # mark claim as used
        claim.mark_used()

        # 4. get evidence that supports and refutes the claim
        evidence = self.kg.supported_by_map[claim]
        counter_evidence = self.kg.refuted_by_map[claim]
        
        return claim, evidence, counter_evidence
    
    def __select_fist_claim(self):
        claim = self.kg.get_most_relative_claim(self.topic)

        claim.mark_used()

        # 4. get evidence that supports and refutes the claim
        evidence = self.kg.supported_by_map[claim]
        counter_evidence = self.kg.refuted_by_map[claim]
        
        return claim, evidence, counter_evidence

    def __format_evidence_list(self, evidence_list: List[EvidenceNode]):
        evidence_list = [evidence.text for evidence in evidence_list]
        return "\n".join(evidence_list)

    def __get_sys_message(self, opponent_msg=None, is_final=False):
        if is_final:
            return { 'role': 'system', 'content': f"""
                IDENTITY and PURPOSE
                    
                You are a debate agent that take a position on the presented topic. 
                You are arguing {self.stance} the topic: '{self.topic}'.
                You are on the final round of the debate.

                INTERNAL ASSISTANT STEPS
                
                Analyze the topic and your previous claims
                List out each claim you made in the debate.
                Then summarize your position on the topic.


                OUTPUT INSTRUCTIONS

                Use bullet points to list out each claim you made in the debate.
                Write in full sentences.

                """}
        
        # if is the first round, select the first claim
        if not opponent_msg:
            claim, evidence, counter_evidence = self.__select_fist_claim()
        else:
            claim, evidence, counter_evidence = self.__select_claim_from_kg(opponent_msg['content'])

        return { 'role': 'system', 'content': f"""
            IDENTITY and PURPOSE
                    
            You are a debate agent that take a position on the presented topic. 
            You are arguing {self.stance} the topic: '{self.topic}'.
            You have access to a claim, supporting evidence, and counter evidence from the knowledge base.
            
            INTERNAL ASSISTANT STEPS

            Analyze the topic and the previous conversation and make sure you dunderstand the current progress of the debate.
            Pay extra attention to the last message from your opponent. As this the the main point you will be responding to.
            Breakdowb the key points from the last opponent message and fomulate your response to refute those points.
            USe the provided claim and evidence in your response to make your argument more convincing.
            
            OUTPUT INSTRUCTIONS

            Do not begin the response with the claim, paraphrase the claim in your own words.
            Be sure to include the provided claim and evidence in your response.
            Avoid using beginning phrases already used in the previous rounds.
            Output all information in one paragraph. 
            Avoid transitional phrases like "in addition" or "furthermore".
            Do not summarize or repeat previous points.
            Limit your response to 2 to 4 sentences.

            LAST OPPONENT MESSAGE:
            {opponent_msg['content']}
            
            CLAIM:
            {claim.text}

            SUPPORTING EVIDENCE:
            {self.__format_evidence_list(evidence)}

            """
            # COUNTER EVIDENCE:
            # {self.__format_evidence_list(counter_evidence)}
        }   

    def next_round_response(self, is_final=False):
        message_history = self.memory_manager.to_msg_array(self.agent_config)
        last_msg = message_history[-1]
        sys_msg = self.__get_sys_message(last_msg, is_final=is_final)
        message_history.insert(0, sys_msg)
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message_history,
            response_model=ResponseModel,
            temperature=1.0
        ),

        if isinstance(resp, ResponseModel):
            self.memory_manager.add_message(self.agent_config, resp.message)
            return resp.message
        if isinstance(resp, tuple):
            self.memory_manager.add_message(self.agent_config, resp[0].message)
            return resp[0].message
        
        raise Exception("Invalid response from OpenAI")

