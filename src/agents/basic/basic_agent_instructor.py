import instructor
from src.debate.basic_history_manager import BasicHistoryManager
from src.shared.models import AgnetConfig, ResponseModel
from src.llm.client import llm
from openai import OpenAI
from typing import Literal

# Patch the OpenAI client
client = instructor.from_openai(llm)

"""
Basic Debate Agent:
- Use message history to generate next round response
- No topic tracking
- No moderator interfereence
"""

class BasicDebateAgent:
    '''
    methods

    1. init topic
    2. init stance (for, against)
    3. next round response
    '''

    def __init__(
            self, 
            topic: str,
            stance: Literal["for", "against"],
            agent_config: AgnetConfig,
            memory_manager: BasicHistoryManager,
            open_ai_client: OpenAI | None = None
        ):
        self.memory_manager = memory_manager
        self.topic = topic
        self.stance = stance
        self.agent_config = agent_config

        # resigter agent
        self.memory_manager.register_agent_debator(agent_config)

    def __get_sys_message(self):
        return [
            {'role': 'system', 'content': f"""
            Background: You are a debate expert. You will be arguing {self.stance} the topic: '{self.topic}'.

            Instructions:
            - Make one point at a time. Limit each response to one point. Each point should only be one to two sentences.
            - Avoid using bullet points or lists. Write in full sentences.
            - Communicate in a conversational tone.
            - Do not summarize or repeat previous points.
            """},
        ]

    def next_round_response(self):
        sys_msgs = self.__get_sys_message()
        message_history = self.memory_manager.to_msg_array(self.agent_config)
        message_history = sys_msgs + message_history
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

    

