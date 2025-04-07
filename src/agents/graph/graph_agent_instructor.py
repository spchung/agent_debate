'''
This agent consumes knowledge base similar to the planning agent.
But instead of making stance summary for each resource, information from all sources are pooled into a knowlege graph

How to build the graph
- iterate over each resource
  1. first read
    - generate claims to support your stance (2 - 3 claims)
    - if claim is similar to claims already generated in previous resouces -> use the same claim 
    - SIMILARITY CHECK (embedding or llm)
  2. second read (for each claim)
    - find evidence that supports or refutes the claim
    - add edge between claim and evidence

Inference:
- extract keypoints from the opponent's response
- find closest claim in the graph 
- find evidence that supports or refutes the claim
'''
 
import instructor
import json
from typing import Literal
from src.llm.client import llm
from src.debate.basic_history_manager import BasicHistoryManager
from src.shared.models import AgnetConfig, ResponseModel
from src.knowledge_base.pdf_kb import PdfKnowledgeBase

class GraphDebateAgnet:
    pass