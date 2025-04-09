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
    