'''
shared utils for openai embedding
'''

import os
from openai import OpenAI

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