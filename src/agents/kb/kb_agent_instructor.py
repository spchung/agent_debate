import instructor
from typing import Literal
from src.llm.client import llm
from src.debate.basic_history_manager import BasicHistoryManager
from src.shared.models import AgnetConfig, ResponseModel
from src.knowledge_base.pdf_kb import PdfKnowledgeBase
from src.agents.kb.workers import ClaimInqueryGeneratorInputSchema, claim_inquery_generator_agent
# Patch the OpenAI client
client = instructor.from_openai(llm)

class KnowledgeBaseDebateAgent:
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

        ## knowledge base
        self.kb = PdfKnowledgeBase(kb_path)
        # self.kb = PdfKnowledgeBase('knowledge_source/quantitative_easing')
    
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
    
    def next_round_response(self, is_final=False):
        # process last response
        opponent_last_msg = self.memory_manager.get_last_message()
        
        ## identify claims
        ## create a question for the knowledge base
        claim_inquery_generator_res = claim_inquery_generator_agent.run(
            ClaimInqueryGeneratorInputSchema(
                topic=self.topic,
                last_response=opponent_last_msg.message if opponent_last_msg else ''
            )
        )
        questions = claim_inquery_generator_res.questions
        
        # retrieval
        retrueved_res = self.kb.query(questions[0])
        context_message = {
            "role": "system",
            "content": f"Consider the following information when responding to the user:\n\n{retrueved_res.response}"
        }
        
        ## REASON

        sys_msg = self.__get_sys_message(is_final=is_final)
        message_history = self.memory_manager.to_msg_array(self.agent_config)
        message_history.insert(0, sys_msg)
        message_history.insert(1, context_message)
        
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
    