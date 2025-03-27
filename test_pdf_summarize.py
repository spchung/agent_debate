import json
from src.utils.pdf_parser import PDFParser
from src.llm.client import llm
from src.agents.planning.workers import OpinionatedTextSummarizer, TextSummarizerInputSchema
from pydantic import BaseModel

f = 'Multiple Structural Breaks in Panel Data'

parser  = PDFParser(f"knowledge_source/quantitative_easing/{f}.pdf",)
def test_load():    

    raw_text = parser.pdf_to_text(
        f"knowledge_source/quantitative_easing/{f}.pdf"
    )
    
    summarizer = OpinionatedTextSummarizer(
        topic='Quantitative Easing is a good policy for long-term economic growth.',
        agent_stance='against'
    )

    res = summarizer.run(
        TextSummarizerInputSchema(
            text=raw_text
        )
    )

    # write to json 
    with open(f"temp.json", 'w') as file:
        file.write(res.model_dump_json())

if __name__ == '__main__':
    test_load()