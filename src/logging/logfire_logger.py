import logfire
import os
from dotenv import load_dotenv

load_dotenv()

logfire.configure(
    token=os.environ.get('LOGFIRE_KEY'),
)