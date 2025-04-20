from pydantic import BaseModel, Field
from src.llm.client import llm
from time import time
from typing import Literal

class IdNameModel(BaseModel):
    id: str
    name: str

class AgnetConfig(IdNameModel):
    type: Literal['debator', 'moderator'] = Field(
        default='debator', description="The type of the agent")

class MessageModel(BaseModel):
    agent_config: AgnetConfig
    message: str
    timestamp: str = Field(default=time(), description="The timestamp of the message")

    def to_dict(self, default_role=None, agent_perspective: AgnetConfig = None):
        
        if not default_role:
            default_role = 'assistant'
        
        # if moderator message - leave as is
        if self.agent_config.type == 'moderator':
            return {
                "role": 'user',
                "content": f"[MODERATOR]: {self.message}"
            }
        
        # if agent message - add [YOU] or [OPPONENT]
        if agent_perspective and agent_perspective.id == self.agent_config.id:
            return {
                "role": default_role,
                "content": f"[YOU]: {self.message}"
            }
        
        return {
            "role": default_role,
            "content": f"[OPPONENT]: {self.message}"
        }

class ResponseModel(BaseModel):
    message: str = Field(..., description="The response of the agent")
