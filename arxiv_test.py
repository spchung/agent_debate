from paper_processing.arxiv_processor import ArxivScraper

scraper = ArxivScraper()

# Just download the PDF
result = scraper.download_and_extract_from_pdf_url("https://arxiv.org/pdf/1508.03395")
print(f"PDF saved to: {result['pdf_path']}")

# Download and extract text (if PyPDF2 is installed)
result = scraper.download_and_extract_from_pdf_url(
    "https://arxiv.org/pdf/1508.03395", 
    extract_text=True
)
print(result)