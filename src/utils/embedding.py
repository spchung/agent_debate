'''
shared utils for openai embedding
'''

import os
from openai import OpenAI
import numpy as np

def get_openai_embedding(text, model="text-embedding-3-small"):
    try:
        # Initialize the client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Make the API call to get the embedding
        response = client.embeddings.create(
            input=text,
            model=model
        )
        
        # Extract the embedding from the response
        embedding = response.data[0].embedding
        
        return embedding  # Returns the embedding as a plain list
    
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must be of the same length")
    
    if isinstance(a, list):
        a = np.array(a)
    
    if isinstance(b, list):
        b = np.array(b)

    # Calculate cosine similarity
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)
