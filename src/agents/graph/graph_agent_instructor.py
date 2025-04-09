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
            logger.info(f"Claim {claim.uuid} already used, finding next most relative claim")
            # find next most relative claim
            claim = self.kg.find_next_relative_claim(claim)
            logger.info(f"now using claim {claim.uuid}")

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
            self_messages = self.memory_manager.get_messages_of_agent(self.agent_config)
            # Extract previous arguments
            my_previous_arguments = []
            for msg in self_messages:
                if 'role' in msg and msg['role'] == 'assistant':
                    my_previous_arguments.append(msg['content'])
            
            previous_args_text = ""
            if my_previous_arguments:
                previous_args_text = "\n".join(my_previous_arguments)
                
            return { 'role': 'system', 'content': f"""
                IDENTITY and PURPOSE
                    
                You are a skilled debate agent taking a position on the presented topic. 
                You are arguing {self.stance} the topic: '{self.topic}'.
                You are on the final round of the debate and need to create a compelling summary.

                INTERNAL ASSISTANT STEPS
                
                1. Carefully analyze all your previous statements in the debate, provided below.
                2. Identify 4-5 key arguments and claims you've consistently made throughout the debate.
                3. Extract the strongest evidence and points from your previous arguments.
                4. Organize these into a logical, coherent structure that reinforces your position.
                5. Create a summary that presents your stance as a unified, well-reasoned argument.

                OUTPUT INSTRUCTIONS

                1. Use bullet points to list out each key claim you've made in the debate.
                2. Format each bullet point with bold headers that capture the essence of each argument.
                3. Under each point, provide 1-2 sentences of explanation drawing from your previous statements.
                4. Ensure your summary presents a logical, interconnected narrative supporting your position.

                YOUR PREVIOUS ARGUMENTS IN THIS DEBATE:
                {previous_args_text}
                """}
        
        # if is the first round, select the first claim
        if not opponent_msg:
            claim, evidence, counter_evidence = self.__select_fist_claim()
        else:
            claim, evidence, counter_evidence = self.__select_claim_from_kg(opponent_msg['content'])

        # Get the debate history to provide context
        message_history = self.memory_manager.to_msg_array(self.agent_config)
        my_previous_arguments = []
        for msg in message_history:
            if msg.get('role') == 'assistant':
                my_previous_arguments.append(msg.get('content', ''))
        
        previous_args_text = ""
        if my_previous_arguments:
            previous_args_text = "YOUR PREVIOUS ARGUMENTS:\n" + "\n".join(my_previous_arguments[-2:] if len(my_previous_arguments) > 2 else my_previous_arguments)

        return { 'role': 'system', 'content': f"""
            IDENTITY and PURPOSE
                    
            You are a skilled debate agent arguing {self.stance} the topic: '{self.topic}'.
            You are participating in a coherent, ongoing debate where each argument builds upon previous points.
            You have access to a claim, supporting evidence, and counter evidence from your knowledge base.
            
            DEBATE STRATEGY
            
            1. Build a cohesive narrative throughout the debate - your arguments should connect to each other
            2. Directly address and counter your opponent's most recent points using specific evidence
            3. Incorporate your selected claim and supporting evidence to strengthen your position
            4. Acknowledge counter evidence but rebut it effectively to demonstrate critical thinking
            5. Reference your previous arguments to show continuity and logical progression
            6. Be strategic - anticipate counter-arguments and address them preemptively
            
            INTERNAL ASSISTANT STEPS

            1. Carefully analyze the entire debate history and identify the main thread of arguments
            2. Pay special attention to your opponent's most recent message and extract their key points
            3. Consider how your new claim relates to or extends your previous arguments
            4. Review the counter evidence and prepare thoughtful rebuttals to it
            5. Formulate a response that maintains your position while directly addressing opponent's points
            6. Integrate the provided claim and supporting evidence to strengthen your argument
            7. Address and refute aspects of the counter evidence to demonstrate critical thinking
            
            OUTPUT INSTRUCTIONS

            - Create a cohesive, persuasive response that builds on your previous arguments
            - Directly address specific points raised by your opponent, citing evidence
            - Acknowledge some aspect of the counter evidence but provide a substantive rebuttal to it
            - Integrate (don't just repeat) the provided claim and supporting evidence
            - Maintain a consistent argumentative stance throughout the debate
            - Use clear, concise language in a natural conversational style
            - Do not start the message with "[YOU]" or "[AGENT]" or any other identifier
            - If an abbreviation has been used in the previous messages, use the same abbreviation and do not repeat the full form
            - Aim for 4-6 sentences that form a cohesive paragraph

            DEBATE CONTEXT:
            {previous_args_text}

            LAST OPPONENT MESSAGE:
            {opponent_msg['content'] if opponent_msg else ""}
            
            CLAIM TO INCORPORATE:
            {claim.text}

            SUPPORTING EVIDENCE:
            {self.__format_evidence_list(evidence)}

            COUNTER EVIDENCE TO ADDRESS AND REBUT:
            {self.__format_evidence_list(counter_evidence)}
            """
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
            temperature=0.8
        ),

        if isinstance(resp, ResponseModel):
            self.memory_manager.add_message(self.agent_config, resp.message)
            return resp.message
        if isinstance(resp, tuple):
            self.memory_manager.add_message(self.agent_config, resp[0].message)
            return resp[0].message
        
        raise Exception("Invalid response from OpenAI")

