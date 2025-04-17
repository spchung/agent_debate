from src.shared.models import AgnetConfig
from src.knowledge_base.pdf_kb import PdfKnowledgeBase
from src.knowledge_base.pdf_kb import PdfKnowledgeBase
from src.debate.basic_history_manager import BasicHistoryManager
from src.agents.kb.kb_agent_instructor import KnowledgeBaseDebateAgent
from dotenv import load_dotenv
load_dotenv()

shared_mem = BasicHistoryManager()
moderator = AgnetConfig(id="moderator", name="Moderator", type="moderator")
shared_mem.register_agent_moderator(moderator)

path = 'knowledge_source/ai_regulation'
kb = PdfKnowledgeBase(path)

topic = "Self-regulation by the AI industry is preferable to government regulation."

agent_2 = KnowledgeBaseDebateAgent(
    topic=topic,
    stance="for",
    agent_config=AgnetConfig(id="agent_2", name="Agent 2"),
    memory_manager=shared_mem,
    kb_path='knowledge_source/ai_regulation',
)

while True:
    query = input("Enter your query: ")
    if query == "exit":
        break
    res = kb.query(query)
    print("RES: ", res.response)
    
    file_name = res.file_name
    source = agent_2.file_to_autor_title_map[file_name]
    print("Source: ", source)