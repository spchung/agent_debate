import requests
import xml.etree.ElementTree as ET
import re, time
from bs4 import BeautifulSoup
import urllib.parse

class ArxivScraper:
    """
    scape arxiv papers ans turn into RAG chunks
    """

