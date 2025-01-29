from pydantic import BaseModel
from typing import List

from schemas.agents_schema import AgentResponse

#Request Models
class CreateTeamRequest(BaseModel):
    name: str
    agent_names: List[str]
    instructions: List[str]

#Response Models
class TeamResponse(BaseModel):
    id: str
    user_id: str
    name: str
    agents: List[AgentResponse]
    instructions: List[str]
    markdown: bool
    show_tool_calls: bool

class CreateTeamResponse(BaseModel):
    message: str
    team: TeamResponse

class RunTeamRequest(BaseModel):
    message: str
    team_id: str
