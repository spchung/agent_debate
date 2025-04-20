import json
from src.utils.pdf_parser import PDFParser
from src.agents.planning.workers import summarize_knowledge_base_with_stance

dir_path = "knowledge_source/quantitative_easing"

def test_load():    

    for_kb = summarize_knowledge_base_with_stance(dir_path, "Quantitative Easing is a good policy for long-term economic growth.", "for")
    against_kb = summarize_knowledge_base_with_stance(dir_path, "Quantitative Easing is a good policy for long-term economic growth.", "against")


    d = {
        "for": for_kb,
        "against": against_kb
    }

    with open("temp.json", "w") as file:
        json.dump(d, file)

if __name__ == '__main__':
    test_load()