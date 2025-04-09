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
    