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

        # DEBUG - dump planned kb
        with open(f"planned_kb/{self.agent_config.name}.json", "w") as f:
            json.dump(self.planned_kb, f, indent=4)
    
    def describe(self):
        return f"Planning agent for topic: {self.topic} with stance: {self.stance}"

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
                
                You are a debate participant delivering your final statement.
                You are arguing {self.stance} the topic: '{self.topic}'.
                
                INTERNAL PREPARATION
                
                1. Scan through your previous statements in this debate.
                2. Pick out your most compelling arguments and evidence.
                3. Identify the main point from your opponent that needs addressing.
                
                CLOSING FORMAT
                
                1. MAIN POINTS RECAP (1 paragraph)
                - Remind the audience of your 2-3 key arguments
                - State them clearly and confidently without excessive detail
                
                2. OPPONENT COUNTER (1 paragraph)
                - Target your opponent's central claim or weakness
                - Explain why their position doesn't hold up
                
                3. FINAL TAKEAWAY (1-3 sentences)
                - Deliver a concise, powerful conclusion
                - Leave the audience with a clear reason to support your position
                
                YOUR PREVIOUS ARGUMENTS IN THIS DEBATE:
                {previous_args_text}
                
                OUTPUT INSTRUCTIONS
                
                - Be direct and straightforward
                - Use everyday language that's easy to understand
                - Keep your statement under 250 words total
                - Maintain a confident but conversational tone
                - Focus on making your position memorable
                - Write in complete paragraphs, not bullet points
                - Don't use debate jargon or overly formal language
            """}

        return {'role': 'system', 'content': f"""
            IDENTITY and PURPOSE
                    
            You are a debate agent that take a position on the presented topic. 
            You are arguing {self.stance} the topic: '{self.topic}'.

            DEBATE STRATEGY
            
            1. Build a cohesive narrative throughout the debate - your arguments should connect to each other
            2. Directly address and counter your opponent's most recent points using specific evidence
            3. Incorporate your selected claim and supporting evidence to strengthen your position
            4. Acknowledge counter evidence but rebut it effectively to demonstrate critical thinking
            5. Reference your previous arguments to show continuity and logical progression
            6. Be strategic - anticipate counter-arguments and address them preemptively

            INTERNAL ASSISTANT STEPS

            1. Analyze the topic and your previous claims.
            2. Respond to your opponent's last response first if applicable.
            3. Once you have responded to your opponent, consider the key points from the knowledge base as you formulate your response.
            4. Your response should either discredit your opponent's claims or support your own claims.
            5. If your response is based on the knowledge base, ensure that you reference the title or author of the source.
            6. Formulate a response that responds to the points made by your oppoenent in the last round if applicable.

            OUTPUT INSTRUCTIONS

            - Create a cohesive, persuasive response that builds on your previous arguments
            - Directly address specific points raised by your opponent, citing evidence
            - Acknowledge some aspect of the counter evidence but provide a substantive rebuttal to it
            - Prioritze a response that draws from the knowledge base
            - When referencing the knowlege base, cite the author or source of the information
            - Maintain a consistent argumentative stance throughout the debate
            - Use clear, concise language in a natural conversational style
            - Do not start the message with "[YOU]" or "[AGENT]" or any other identifier
            - If an abbreviation has been used in the previous messages, use the same abbreviation and do not repeat the full form
            - Aim for 3 to 5 sentences that form a cohesive paragraph

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
    