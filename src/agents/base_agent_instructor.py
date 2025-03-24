import instructor
from src.debate.history_manager import DebateChatHistoryManager
from src.shared.models import AgnetConfig, ResponseModel
from src.llm.client import llm
from openai import OpenAI
from typing import Literal

# Patch the OpenAI client
client = instructor.from_openai(llm)

class InstructorBaseAgent:
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
            memory_manager: DebateChatHistoryManager,
            open_ai_client: OpenAI | None = None
        ):
        self.memory_manager = memory_manager
        self.topic = topic
        self.stance = stance
        self.agent_config = agent_config

        # resigter agent
        self.memory_manager.register_agent_debator(agent_config)
    
    def __show_message_history(self):
        message_history = self.memory_manager.to_msg_array(self.agent_config)
        print(message_history)

    def __get_sys_message(self):
        return {
            'role': 'system',
            'content': f"You are a debate expert. You will be arguing {self.stance} the topic: '{self.topic}'."
        }

    def next_round_response(self):
        sys_msg = self.__get_sys_message()
        message_history = self.memory_manager.to_msg_array(self.agent_config)
        message_history.insert(0, sys_msg)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message_history,
            response_model=ResponseModel,
        ),

        if isinstance(resp, ResponseModel):
            self.memory_manager.add_message(self.agent_config, resp.message)
            return resp.message
        if isinstance(resp, tuple):
            self.memory_manager.add_message(self.agent_config, resp[0].message)
            return resp[0].message
        
        raise Exception("Invalid response from OpenAI")

    

