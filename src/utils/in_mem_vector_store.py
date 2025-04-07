from docarray import BaseDoc, DocList
from docarray.index import InMemoryExactNNIndex
import os
from openai import OpenAI
from docarray.typing import NdArray
from pydantic import Field
import numpy as np

class ClaimDoc(BaseDoc):
    text: str 
    embedding: NdArray[1536]
    metadata: dict = Field(default_factory=dict)

# Your existing embedding function
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
        
        return embedding
    
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

class InMemoryVectorStore:
    def __init__(self, embedding_dim=1536):
        # Initialize in-memory index with dimension matching OpenAI embeddings
        self.embedding_dim = embedding_dim
        self.documents = DocList[ClaimDoc]()
        self.index = InMemoryExactNNIndex[ClaimDoc]()
        
    def add(self, text, embedding=None, metadata=None):
        """Add a document to the vector store"""
        # Generate embedding
        
        if not embedding:
            embedding = get_openai_embedding(text)
        
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        # Create document with embedding
        doc = ClaimDoc(
            text=text,
            embedding=embedding,
            metadata=metadata or {}
        )

        # Add document to the list
        self.documents.append(doc)
            
        # Add to index
        self.index.index([doc])
        return doc.id
    
    def search(self, query, limit=5):
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = get_openai_embedding(query)

        # Convert the list to a numpy array
        query_embedding_array = np.array(query_embedding)
        
        # Perform search
        results = self.index.find(query_embedding_array, search_field='embedding', limit=limit)
        
        # Format results
        return [
            {
                'id': doc.id,
                'text': doc.text,
                'score': score,
                'metadata': doc.metadata
            }
            for doc, score in zip(results.documents, results.scores)
        ]
    
    def get_by_id(self, doc_id):
        """Retrieve a document by ID"""
        # This would need to be implemented based on how you store document IDs
        # The current DocArray doesn't have a direct ID lookup without additional indexing
        pass
        
    def clear(self):
        """Clear all documents from the store"""
        self.index = InMemoryExactNNIndex[ClaimDoc](dim=self.embedding_dim)
    
    def size(self):
        """Return the number of documents in the store"""
        return len(self.index)