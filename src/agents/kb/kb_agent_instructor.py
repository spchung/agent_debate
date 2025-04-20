import instructor
import os, json
from typing import Literal
from src.llm.client import get_llm_instnace
from src.utils.pdf_parser import PDFParser
from src.shared.models import AgnetConfig, ResponseModel
from src.knowledge_base.pdf_kb import PdfKnowledgeBase
from src.debate.basic_history_manager import BasicHistoryManager
from src.agents.kb.workers import ClaimInqueryGeneratorInputSchema, claim_inquery_generator_agent
from src.agents.kb.workers import title_and_author_extractor_agent, TitleAndAuthorExtractorInputSchema
from src.agents.prompting import closing_remark_prompt

class KnowledgeBaseDebateAgent:
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

        ## knowledge base
        self.kb = PdfKnowledgeBase(kb_path)
        self.file_to_autor_title_map = self.__build_file_name_to_author_map(kb_path)
    
    def describe(self):
        return f"Knowledge Base agent for topic: {self.topic} with stance: {self.stance}"

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
    
    def __get_sys_message(self, is_final=False, context={}, is_opening=False):
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
                
                1. Establish a clear position based on factual evidence
                2. Present your strongest evidence-backed point first
                3. Set the foundation for a fact-based discussion
                4. Reference specific information from your knowledge base
                
                INTERNAL ASSISTANT STEPS
                
                1. Analyze the topic thoroughly.
                2. Select the most compelling keypoint from your knowledge base that supports your position.
                3. Identify the source (title or author) of this keypoint.
                4. Craft an opening statement that clearly presents this evidence-backed position.
                
                OUTPUT INSTRUCTIONS
                
                - Create a compelling opening statement that clearly establishes your evidence-based position
                - Reference specific information from the knowledge base, citing the author or source
                - Use clear, concise language in a natural conversational style
                - Do not start the message with "[YOU]" or "[AGENT]" or any other identifier
                - If using abbreviations, define them on first use
                - Aim for 3 to 5 sentences that form a cohesive paragraph
                - Focus on establishing credibility through factual evidence
                
                KNOWLEDGE BASE
                {json.dumps(context)}
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
    
    def debate_identifier(self):
        return f"kb_{self.stance}"

    def next_round_response(self, is_final=False, is_opening=False):
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

        sys_msg = self.__get_sys_message(is_final=is_final, context=context_dict, is_opening=is_opening)
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
    