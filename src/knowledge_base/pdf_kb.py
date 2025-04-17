from typing import List
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

class RetrievedNodeModel(BaseModel):
    node_id: str
    score: float
    text: str
    source_file_name: str

    def __repr__(self):
        return str(RetrievedNodeModel(
            node_id=self.node_id,
            score=self.score,
            text=self.text[:50] + "..." if len(self.text) > 50 else self.text,
            source_file_name=self.source_file_name
        ))

class QueryResponseModel(BaseModel):
    response: str
    file_name: str
    source_nodes: List[RetrievedNodeModel]

class RetrieveResponseModel(BaseModel):
    nodes: List[RetrievedNodeModel]

class PdfKnowledgeBase:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.__build_index()
        self.__build_query_engine()
        self.__build_retriever()
    
    def __build_index(self):
        documents = SimpleDirectoryReader(self.data_dir).load_data()
        self.index = VectorStoreIndex.from_documents(documents)
    
    def __build_query_engine(self):
        self.query_engine = self.index.as_query_engine(
            response_mode="tree_summarize",
            verbose=True,
            similarity_top_k=1
        )
    
    def __build_retriever(self):
        self.retriever = self.index.as_retriever(
            retriever_mode="llm",
            verbose=True,
            similarity_top_k=1
        )
    
    def query(self, query: str):
        response = self.query_engine.query(query)
        source_nodes = []
        for source_node in response.source_nodes:
            # print(source_node.node.metadata)
            file_name = source_node.node.metadata['file_name']
            source_nodes.append(RetrievedNodeModel(
                node_id=source_node.node_id,
                score=source_node.score,
                text=source_node.text,
                source_file_name=file_name
            ))
        
        return QueryResponseModel(
            response=response.response,
            file_name=response.source_nodes[0].node.metadata['file_name'],
            source_nodes=source_nodes
        )

    def retrieve(self, query: str):
        nodes =  self.retriever.retrieve(query)
        res = []
        for node in nodes:
            file_name = node.node.metadata.get('file_name', 'Unknown Document')
            res.append(RetrievedNodeModel(
                node_id=node.node_id,
                score=node.score,
                text=node.text,
                source_file_name=file_name
            ))
        return RetrieveResponseModel(nodes=res)
    

# if __name__ == "__main__":
#     kb = PdfKnowledgeBase('knowledge_source/quantitative_easing')
#     # resp = kb.retrieve("what is the point of quantitative easing?")
#     # print(resp)
#     resp = kb.query("what is the point of quantitative easing?")
#     print(resp)