import instructor
from typing import Literal
from src.llm.client import llm
from src.debate.basic_history_manager import BasicHistoryManager
from src.shared.models import AgnetConfig, ResponseModel

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
    1. init topic
    2. init stance (for, against)
    3. next round response
    '''

    def __init__(self, topic: str, stance: Literal["for", "against"], agent_config: AgnetConfig, memory_manager: BasicHistoryManager):
        self.memory_manager = memory_manager
        self.topic = topic
        self.stance = stance
        self.agent_config = agent_config

        # resigter agent
        self.memory_manager.register_agent_debator(agent_config)

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
                
                You are a debate agent making your final closing statement.
                You are arguing {self.stance} the topic: '{self.topic}'.
                
                INTERNAL PROCESS
                
                1. Review your previous arguments in this debate.
                2. Identify your 2-3 strongest points.
                3. Recognize your opponent's main argument that needs addressing.
                
                CLOSING STATEMENT STRUCTURE
                
                1. QUICK SUMMARY (1-2 paragraphs)
                - Briefly recap your 2-3 strongest arguments
                - Connect these points to show why your position is correct
                
                2. SIMPLE REBUTTAL (1 paragraph)
                - Address your opponent's most significant point
                - Explain why their argument is flawed or insufficient
                
                3. MEMORABLE CLOSING (1-2 sentences)
                - End with a clear, impactful statement supporting your position
                - Make it something the audience will remember
                
                YOUR PREVIOUS ARGUMENTS IN THIS DEBATE:
                {previous_args_text}
                
                OUTPUT INSTRUCTIONS
                
                - Keep it brief and to the point (under 300 words total)
                - Use conversational language, not formal debate terminology
                - Focus on clarity and impact rather than elaborate rhetoric
                - Write in flowing paragraphs, not bullet points
                - Don't introduce completely new arguments
                - Speak directly to the audience in a confident tone
            """}

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

    

