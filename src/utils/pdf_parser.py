## atomic agents for specific tasks
import instructor
from typing import List
from pydantic import Field
from src.llm.client import llm
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from llmsherpa.readers import LayoutPDFReader

'''
1. article summarizer
'''
import PyPDF2
import re
import os
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFParser:
    '''
    Extract sections from a given pdf file (published paper) and produce a llama Document object 
    '''
    def __init__(self):
        pass

    def pdf_to_text(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        logger.info(f"Extracting text from {pdf_path}")
        text = ""
        
        try:
            # Open the PDF file in read-binary mode
            with open(pdf_path, 'rb') as file:
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Get the number of pages
                num_pages = len(pdf_reader.pages)
                logger.info(f"PDF has {num_pages} pages")
                
                # Extract text from each page
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text += page_text + "\n\n"
                    
            logger.info(f"Successfully extracted {len(text)} characters of text")
            return text
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def auto_detect_sections(self, text: str) -> List[Tuple[str, int]]:
        """
        Automatically detect potential section headers in the text.
        
        Args:
            text (str): Extracted text from PDF
            
        Returns:
            List[Tuple[str, int]]: List of tuples containing section titles and their positions
        """
        logger.info("Auto-detecting section headers")
        
        # Common section header patterns
        patterns = [
            # Numbered sections like "1. Introduction", "1.1 Background", etc.
            r"^(\d+\.(?:\d+)*\s+[A-Z][A-Za-z0-9\s\-:,]+)$",
            
            # Capitalized headers like "INTRODUCTION", "LITERATURE REVIEW", etc.
            r"^([A-Z][A-Z\s\-:,]+)$",
            
            # Title case headers like "Introduction", "Literature Review", etc.
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)$",
            
            # Headers with a colon like "Abstract:", "Method:", etc.
            r"^([A-Z][A-Za-z\s\-]+):$",
            
            # Markdown-style headers (# Title, ## Subtitle, etc.)
            r"^(#{1,6}\s+[A-Za-z0-9\s\-:,]+)$",

            ## manually match 'abstract'
            r"^(Abstract)$"
        ]
        
        # Find all potential section headers
        section_matches = []
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                section_title = match.group(1).strip()
                start_pos = match.start()
                section_matches.append((section_title, start_pos))
        
        # Sort matches by position in text
        section_matches.sort(key=lambda x: x[1])
        
        # Filter out likely false positives
        filtered_matches = []
        prev_pos = -1
        
        for title, pos in section_matches:
            # Skip if too close to previous match (likely part of the same section)
            if prev_pos != -1 and pos - prev_pos < 50:
                continue
            
            # Skip very short lines unless they are numbered sections
            if len(title) < 5 and not re.match(r"^\d+", title):
                continue
                
            filtered_matches.append((title, pos))
            prev_pos = pos
        
        logger.info(f"Detected {len(filtered_matches)} potential section headers")
        return filtered_matches

    def analyze_section_structure(self, section_matches: List[Tuple[str, int]], text: str) -> Dict[str, List[Tuple[str, int]]]:
        """
        Analyze the structure of detected sections to identify hierarchy.
        
        Args:
            section_matches (List[Tuple[str, int]]): List of section titles and positions
            text (str): Full text content
            
        Returns:
            Dict[str, List[Tuple[str, int]]]: Dictionary grouping sections by their type
        """
        logger.info("Analyzing section structure")
        
        # Group sections by their pattern type
        section_types = {
            "numbered": [],  # Numbered sections (1., 1.1, etc.)
            "capitalized": [],  # ALL CAPS
            "titlecase": [],  # Title Case
            "other": []  # Other formats
        }
        
        for title, pos in section_matches:
            # Check if it's a numbered section
            if re.match(r"^\d+\.(?:\d+)*\s+", title):
                section_types["numbered"].append((title, pos))
            # Check if it's ALL CAPS
            elif title.isupper():
                section_types["capitalized"].append((title, pos))
            # Check if it's Title Case
            elif all(word[0].isupper() for word in title.split() if word and word[0].isalpha()):
                section_types["titlecase"].append((title, pos))
            # Other format
            else:
                section_types["other"].append((title, pos))
        
        # Log counts for each type
        for type_name, sections in section_types.items():
            logger.info(f"Found {len(sections)} {type_name} sections")
        
        return section_types

    def extract_sections_from_matches(self, section_matches: List[Tuple[str, int]], text: str) -> Dict[str, str]:
        """
        Extract section content based on detected section headers.
        
        Args:
            section_matches (List[Tuple[str, int]]): List of section titles and positions
            text (str): Full text content
            
        Returns:
            Dict[str, str]: Dictionary mapping section names to section content
        """
        logger.info("Extracting content for each section")
        
        # Extract content between section headers
        sections = {}
        
        for i, (section_title, start_pos) in enumerate(section_matches):
            # Find the end of the current section (start of next section or end of text)
            if i < len(section_matches) - 1:
                end_pos = section_matches[i + 1][1]
            else:
                end_pos = len(text)
            
            # Find the end of the header line
            header_end = text.find('\n', start_pos)
            if header_end == -1:  # If no newline found
                header_end = start_pos + len(section_title)
            else:
                header_end += 1  # Include the newline
            
            # Extract section content (skip the header itself)
            section_content = text[header_end:end_pos].strip()
            
            # Clean up the section title (remove any markdown formatting, etc.)
            clean_title = re.sub(r'^#+\s+', '', section_title)
            
            sections[clean_title] = section_content
            logger.info(f"Extracted section: {clean_title} ({len(section_content)} chars)")
        
        return sections

    def analyze_pdf_structure(self, pdf_path: str) -> Dict[str, str]:
        """
        Analyze PDF structure to identify and extract all sections.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict[str, str]: Dictionary mapping section names to section content
        """
        # Extract text from PDF
        text = self.pdf_to_text(pdf_path)
        
        if not text:
            logger.error("Failed to extract text from PDF")
            return {}
        
        # Auto-detect section headers
        section_matches = self.auto_detect_sections(text)
        
        # Analyze section structure
        section_types = self.analyze_section_structure(section_matches, text)
        
        # Determine the most likely section format
        primary_format = max(section_types.items(), key=lambda x: len(x[1]))[0]
        logger.info(f"Primary section format appears to be: {primary_format}")

        # Use the primary format sections by default, but fall back to others if few are found
        if len(section_types[primary_format]) >= 3:
            logger.info(f"Using {len(section_types[primary_format])} {primary_format} sections")
            selected_matches = section_types[primary_format]
        else:
            # If primary format has too few sections, use all detected sections
            logger.info("Using all detected sections due to low count in primary format")
            selected_matches = section_matches
        
        # Extract content for each section
        sections = self.extract_sections_from_matches(selected_matches, text)
        
        return sections

    def save_sections_to_files(self, sections: Dict[str, str], output_dir: str) -> None:
        """
        Save extracted sections to individual text files.
        
        Args:
            sections (Dict[str, str]): Dictionary mapping section names to section content
            output_dir (str): Directory to save the section files
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Save each section to a file
        for section_title, content in sections.items():
            # Create a safe filename from the section title
            safe_filename = re.sub(r'[^\w\s-]', '', section_title).strip().replace(' ', '_')
            file_path = os.path.join(output_dir, f"{safe_filename}.txt")
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            
            logger.info(f"Saved section '{section_title}' to {file_path}")
        
        logger.info(f"All {len(sections)} sections saved to {output_dir}")

    def print_section_outline(self, sections: Dict[str, str]) -> None:
        """
        Print a formatted outline of all detected sections.
        
        Args:
            sections (Dict[str, str]): Dictionary mapping section names to section content
        """
        print("\n----- PDF SECTION OUTLINE -----")
        print(f"Total sections detected: {len(sections)}")
        print("----------------------------")
        
        for i, (title, content) in enumerate(sections.items(), 1):
            word_count = len(content.split())
            print(f"{i}. {title} ({word_count} words)")
        
        print("----------------------------\n")

    def extract_pdf_sections(self, pdf_path: str, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Main function to extract all sections from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (Optional[str]): Directory to save sections (if None, won't save)
            
        Returns:
            Dict[str, str]: Dictionary mapping section names to section content
        """
        logger.info(f"Starting PDF section extraction for: {pdf_path}")
        
        # Analyze PDF and extract sections
        sections = self.analyze_pdf_structure(pdf_path)
        
        # Print outline of all detected sections
        self.print_section_outline(sections)
        
        # Save sections to files if output_dir is provided
        if output_dir and sections:
            self.save_sections_to_files(sections, output_dir)
        
        logger.info("PDF section extraction complete")
        return sections