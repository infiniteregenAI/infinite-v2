from pydantic import BaseModel
from typing import List, Optional

# Request Models
class CreateAgentRequest(BaseModel):
    name: str
    role: str
    tools: List[str]
    description: str
    instructions: List[str]
    urls: Optional[List[str]] = []

class UpdateAgentRequest(BaseModel):
    name: str
    role: str
    tools: List[str]
    description: str
    instructions: List[str]
    urls: Optional[List[str]] = []

# Response Models
class AgentResponse(BaseModel):
    id: str 
    user_id: str
    name: str
    role: str
    tools: List[str]
    description: str
    instructions: List[str]
    urls: List[str]
    markdown: bool
    show_tool_calls: bool
    add_datetime_to_instructions: bool

class CreateAgentResponse(BaseModel):
    message: str
    agent: AgentResponse

class UserAgentsResponse(BaseModel):
    agents: List[AgentResponse]

class UpdateAgentResponse(BaseModel):
    message: str
    agent: AgentResponse

class RunAgentRequest(BaseModel):
    message: str
    agent_id: str

class DeleteAgentResponse(BaseModel):
    message: str
    body: AgentResponse
