'''
Planning agent workflow:

1. iterate over each provided resource in kb and make summarizations of the resource
2. make a saummary and list of key points in each resource
3. generate points from summaries and key points that can be used in the debate
4. re rank the points based on the relevance to the topic and strength
'''

import instructor
import json
from typing import Literal
from src.llm.client import llm
from src.debate.basic_history_manager import BasicHistoryManager
from src.shared.models import AgnetConfig, ResponseModel
from src.knowledge_base.pdf_kb import PdfKnowledgeBase
from src.agents.planning.workers import get_kb_with_stance_as_dict

client = instructor.from_openai(llm)

class PlanningDebateAgent:
    def __init__(
        self,
        topic: str,
        stance: Literal["for", "against"],
        agent_config: AgnetConfig,
        memory_manager: BasicHistoryManager,
        kb_path: str
    ):
        self.memory_manager = memory_manager
        self.topic = topic
        self.stance = stance
        self.agent_config = agent_config

        # resigter agent
        self.memory_manager.register_agent_debator(agent_config)

        # planned kb 
        self.planned_kb = get_kb_with_stance_as_dict(kb_path, topic, stance) # List[dict]
    
    def __get_sys_message(self, is_final=False):
        if is_final:
            return {'role': 'system', 'content': f"""
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
                """
                }

        return {'role': 'system', 'content': f"""
            IDENTITY and PURPOSE
                    
            You are a debate agent that take a position on the presented topic. 
            You are arguing {self.stance} the topic: '{self.topic}'.

            INTERNAL ASSISTANT STEPS

            Analyze the topic and your previous claims.
            Respond to your opponent's last response first if applicable.
            Once you have responded to your opponent, consider the key points from the knowledge base as you formulate your response.
            Your response should either discredit your opponent's claims or support your own claims.
            Consider the key points from the knowledge base as you formulate your response.
            Formulate a response that responds to the points made by your oppoenent in the last round if applicable.

            OUTPUT INSTRUCTIONS

            No need to repeat the topic or the last response.
            Use the key points from the knowledge base to support your response.
            Use data points and evidence to support your claims if possible.
            If a source from the knowledge base is used, cite the author name or the title of the source.
            Output all information in one paragraph. 
            Do not use transitional phrases like "in addition" or "furthermore".
            Do not summarize or repeat previous points.
            Limit your response to 5 to 7 sentences.


            KNOWLEDGE BASE

            {json.dumps(self.planned_kb)}
            """
        }
    
    def next_round_response(self, is_final=False):
        sys_msg = self.__get_sys_message(is_final=is_final)
        message_history = self.memory_manager.to_msg_array(self.agent_config)
        message_history.insert(0, sys_msg)
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message_history,
            response_model=ResponseModel,
            temperature=0.7
        ),

        if isinstance(resp, ResponseModel):
            self.memory_manager.add_message(self.agent_config, resp.message)
            return resp.message
        if isinstance(resp, tuple):
            self.memory_manager.add_message(self.agent_config, resp[0].message)
            return resp[0].message
        
        raise Exception("Invalid response from OpenAI")
    