from pydantic import BaseModel
from typing import List, Optional, Any, Dict 
from fastapi import Form, File

from fastapi import UploadFile


class AgentCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    role: Optional[str] = None
    instructions: Optional[List[str]] = None
    tools: Optional[List[str]] = None
    pdf_urls: Optional[List[str]] = None
    website_urls: Optional[List[str]] = None

class AgentModel(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None

class AgentGetResponse(BaseModel):
    agent_id: str
    name: Optional[str] = None
    model: Optional[AgentModel] = None
    add_context: Optional[bool] = None
    tools: Optional[List[Dict[str, Any]]] = None
    memory: Optional[Dict[str, Any]] = None
    storage: Optional[Dict[str, Any]] = None
    knowledge: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    instructions: Optional[List[str]] = None

class AgentRunRequest(BaseModel):
    message: str
    agent_id: str
    stream: bool = True
    monitor: bool = False
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    files: Optional[List[UploadFile]] = None
    
class AgentRenameRequest(BaseModel):
    name: str
    agent_id: str
    session_id: str

class AgentSessionDeleteRequest(BaseModel):
    agent_id: str
    session_id: str

class AgentSessionsRequest(BaseModel):
    agent_id: str

class AgentSessionsResponse(BaseModel):
    title: Optional[str] = None
    session_id: Optional[str] = None
    session_name: Optional[str] = None
    created_at: Optional[int] = None

class WorkflowSessionsRequest(BaseModel):
    user_id: Optional[str] = None

class WorkflowRenameRequest(BaseModel):
    name: str

class WorkflowRunRequest(BaseModel):
    input: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class AgentRunRequest(BaseModel):
    message: str = Form(...),
    agent_id: str = Form(...),
    stream: bool = Form(True),
    monitor: bool = Form(False),
    session_id: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    image: Optional[UploadFile] = File(None)