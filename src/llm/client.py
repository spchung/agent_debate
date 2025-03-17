from src.logging.logfire_logger import logfire
import os
from openai import OpenAI

API_KEY = ""
if not API_KEY:
    API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "API key is not set. Please set the API key as a static variable or in the environment variable OPENAI_API_KEY."
    )

llm = OpenAI(api_key=API_KEY)
logfire.instrument_openai(llm)