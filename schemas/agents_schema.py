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
    name: Optional[str] = None
    role: Optional[str] = None
    tools: Optional[List[str]] = None
    description: Optional[str] = None
    instructions: Optional[List[str]] = None
    pdf_urls: Optional[List[str]] = []
    website_urls: Optional[List[str]] = []  

class AgentResponse(BaseModel):
    id: str 
    user_id: str
    name: str
    role: str
    tools: List[str]
    description: str
    instructions: List[str]
    pdf_urls: List[str]  
    website_urls: List[str]  
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
