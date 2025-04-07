from src.agents.graph.workers import claim_extraction_agent, ResourceClaimExtractionInputSchema
from src.utils.pdf_parser import PDFParser

file = '/Users/stephen/Nottingham/arbitrary_arbitration/knowledge_source/quantitative_easing/qe_turbulance.pdf'
TOPIC = "Quantitative Easing is a good policy for long-term economic growth."

if __name__ == '__main__':
    # Parse the PDF file
    pdf_parser = PDFParser()
    
    # Get the text from the PDF file
    raw_text = pdf_parser.pdf_to_text(file)
    
    # Get the title and author from the PDF file
    # title = pdf_parser.get_title()
    # author = pdf_parser.get_author()

    # Extract claims from the text
    claims = claim_extraction_agent.run(
        ResourceClaimExtractionInputSchema(
            resource_raw_tex=raw_text,
            topic=TOPIC,
            stance='against'
        )
    )
    
    print(claims)

