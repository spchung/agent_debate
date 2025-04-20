import instructor
from typing import Literal
from src.llm.client import get_llm_instnace
from src.debate.basic_history_manager import BasicHistoryManager
from src.shared.models import AgnetConfig, ResponseModel
from src.agents.prompting import closing_remark_prompt

# Patch the OpenAI client
client = instructor.from_openai(get_llm_instnace())

"""
Basic Debate Agent:
- Use message history to generate next round response
- No topic tracking
- No moderator interfereence
"""

class BasicDebateAgent:
    '''
    1. init topic
    2. init stance (for, against)
    3. next round response
    '''

    def __init__(self, 
            topic: str, 
            stance: Literal["for", "against"], 
            agent_config: AgnetConfig, 
            memory_manager: BasicHistoryManager = None
        ):
        self.memory_manager = memory_manager
        self.topic = topic
        self.stance = stance
        self.agent_config = agent_config

        # resigter agent
        # self.memory_manager.register_agent_debator(agent_config)

    def __get_sys_message(self, is_final=False, is_opening=False):
        if is_final:
            self_messages = self.memory_manager.get_messages_of_agent(self.agent_config)
            return closing_remark_prompt(
                stance=self.stance,
                topic=self.topic,
                messages=self_messages,
            )

        if is_opening:
            return {'role': 'system', 'content': f"""
                IDENTITY and PURPOSE
                        
                    You are a debate agent that takes a position on the presented topic. 
                    You are arguing {self.stance} the topic: '{self.topic}'.

                    INTERNAL ASSISTANT STEPS
                    
                    Analyze the topic carefully.
                    Consider the strongest initial argument for your position.
                    Prepare a clear and concise opening statement.

                    OUTPUT INSTRUCTIONS

                    Present a strong opening position on the topic.
                    Make one clear point that establishes your stance. Keep it to one to two sentences.
                    Avoid using bullet points or lists. Write in full sentences.
                    Communicate in a conversational tone.
                    Be direct and confident in establishing your position.
                    Do not preemptively address potential counter-arguments.
                """
            }

        return {'role': 'system', 'content': f"""
            IDENTITY and PURPOSE
                    
                You are a debate agent that take a position on the presented topic. 
                You are arguing {self.stance} the topic: '{self.topic}'.

                INTERNAL ASSISTANT STEPS
                
                Analyze the topic and the previous response from your opponent.
                Use the information provided to generate a response.
                Identify claims from your opponent's last response and address them directly


                OUTPUT INSTRUCTIONS

                No need to repeat the topic or the last response.
                Make one point at a time. Limit each response to one point. Each point should only be one to two sentences.
                Avoid using bullet points or lists. Write in full sentences.
                Communicate in a conversational tone.
                Do not summarize or repeat previous points.
            """
        }

    def debate_identifier(self):
        return f"basic_{self.stance}"

    def describe(self):
        return f"Basic Debate Agent: {self.agent_config.name} - {self.topic} - {self.stance}"

    def next_round_response(self, is_final=False, is_opening=False):
        sys_msg = self.__get_sys_message(is_final=is_final, is_opening=is_opening)
        message_history = self.memory_manager.to_msg_array(self.agent_config)
        message_history.insert(0, sys_msg)
        if is_final:
            resp = client.completions.create(
                model="gpt-4o-mini",
                messages=[sys_msg],
                response_model=ResponseModel,
                temperature=0.3
            )
        else:
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

    

