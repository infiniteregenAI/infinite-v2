from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from schemas.agents_schema import AgentResponse

class CreateTeamRequest(BaseModel):
    name: str
    description: Optional[str] = None
    role: Optional[str] = None
    instructions: Optional[List[str]] = []
    tools: Optional[List[str]] = []
    agent_ids: Optional[List[str]] = []

class TeamResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    role: Optional[str]
    instructions: List[str]
    tools: List[str]
    owner_id: str
    agents: List[AgentResponse]
    is_active: bool
