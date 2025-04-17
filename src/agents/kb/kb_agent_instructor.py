import instructor
import os, json
from typing import Literal
from src.llm.client import llm
from src.debate.basic_history_manager import BasicHistoryManager
from src.shared.models import AgnetConfig, ResponseModel
from src.knowledge_base.pdf_kb import PdfKnowledgeBase
from src.agents.kb.workers import ClaimInqueryGeneratorInputSchema, claim_inquery_generator_agent
from src.agents.kb.workers import title_and_author_extractor_agent, TitleAndAuthorExtractorInputSchema
from src.utils.pdf_parser import PDFParser
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
        self.file_to_autor_title_map = self.__build_file_name_to_author_map(kb_path)
    
    def __build_file_name_to_author_map(self, kb_path: str):
        mmap = {} # file_name -> {author:"", title:""}
        parser = PDFParser()
        for file in os.listdir(kb_path):
            if not file.endswith(".pdf"):
                continue

            raw_text = parser.pdf_to_text(f"{kb_path}/{file}")
            cut_raw_text = raw_text[:2000]    

            result = title_and_author_extractor_agent.run(
                TitleAndAuthorExtractorInputSchema(
                    text=cut_raw_text
                )
            )
            
            mmap[file] = {
                "author": result.author,
                "title": result.title
            }
        
        return mmap
    
    def __get_sys_message(self, is_final=False, context={}):
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
                3. Your response should either discredit your opponent's claims or support your own claims.
                4. You must use the 'keypoint' provided in the knowledge base to support your claims.
                5. Make sure to include the source (title or author) of the keypoint in your response.
                5. Formulate a response that responds to the points made by your oppoenent in the last round if applicable.


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

                {json.dumps(context)}
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
        retrieval = self.kb.query(questions[0])
        source = self.file_to_autor_title_map[retrieval.file_name]
        retrieval_response = retrieval.response
        
        context_dict = {
            "keypoint": retrieval_response,
            "source": source
        }

        sys_msg = self.__get_sys_message(is_final=is_final, context=context_dict)
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
    