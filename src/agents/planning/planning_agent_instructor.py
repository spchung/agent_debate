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
from src.llm.client import get_llm_instnace
from src.debate.basic_history_manager import BasicHistoryManager
from src.shared.models import AgnetConfig, ResponseModel
from src.agents.planning.workers import get_kb_with_stance_as_dict
from src.agents.prompting import closing_remark_prompt

class PlanningDebateAgent:
    def __init__(
        self,
        topic: str,
        stance: Literal["for", "against"],
        agent_config: AgnetConfig,
        kb_path: str,
        memory_manager: BasicHistoryManager = None,
    ):
        self.memory_manager = memory_manager
        self.topic = topic
        self.stance = stance
        self.agent_config = agent_config

        # resigter agent
        # self.memory_manager.register_agent_debator(agent_config)

        # planned kb 
        self.planned_kb = get_kb_with_stance_as_dict(kb_path, topic, stance) # List[dict]

        # DEBUG - dump planned kb
        with open(f"planned_kb/{self.agent_config.name}.json", "w") as f:
            json.dump(self.planned_kb, f, indent=4)
    
    def describe(self):
        return f"Planning agent for topic: {self.topic} with stance: {self.stance}"

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

                OPENING STATEMENT STRATEGY
                1. Introduce your position clearly and confidently
                2. Outline one key argument that will form the foundation of your case
                3. Present your strongest evidence upfront to establish credibility
                4. Frame the debate in terms favorable to your position

                INTERNAL ASSISTANT STEPS
                1. Analyze the topic and your stance thoroughly.
                2. Identify your strongest argument from the knowledge base.
                3. Structure your opening statement to support this argument.
                4. Include relevant evidence from the knowledge base to support each point.
                5. If your opening is based on the knowledge base, ensure that you reference the title or author of the source.
                6. Craft a compelling conclusion that reinforces your main position.

                OUTPUT INSTRUCTIONS
                - Do not include any personal identifiers or greetings
                - Create a strong, persuasive opening statement that clearly establishes your position
                - Include specific evidence to support each argument, citing sources when using the knowledge base
                - Use clear, concise language in a natural conversational style
                - Do not start the message with "[YOU]" or "[AGENT]" or any other identifier
                - If using abbreviations, define them on first use
                - Aim for 4-5 sentences that form a cohesive opening statement

                KNOWLEDGE BASE
                {json.dumps(self.planned_kb)}
                """
                }

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
    
    def next_round_response(self, is_final=False, is_opening=False):
        sys_msg = self.__get_sys_message(is_final=is_final, is_opening=is_opening)
        message_history = self.memory_manager.to_msg_array(self.agent_config)
        message_history.insert(0, sys_msg)
        
        client = instructor.from_openai(get_llm_instnace())
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
    