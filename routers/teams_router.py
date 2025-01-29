import traceback
import logging
from typing import List
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
 
from schemas.teams_schema import CreateTeamRequest, CreateTeamResponse , TeamResponse , RunTeamRequest
from utils.agent_manager import AgentManager

agent_manager = AgentManager()
router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/create-team", response_model=CreateTeamResponse)
async def create_team(request: Request):
    """
        This endpoint creates a new team.
        
        Args:
            request (Request): The request body.
            
        Returns:
            CreateTeamResponse: The response body.
    """
    try:
        
        body = await request.body()
        json_body = json.loads(body)
        request_body = CreateTeamRequest(**json_body)
        
        logger.info(f"Creating team with name: {request_body.name}")
        
        team = agent_manager.create_team(
            name=request_body.name,
            user_id=request.state.user.get("sub"),
            agent_names=request_body.agent_names,
            instructions=request_body.instructions,
        )
        return CreateTeamResponse(
            message="Team created successfully",
            team=team
        )
        
    except ValueError as e:
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=400,
            content={"message": str(e)}
        )
        
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error"}
        )
        
@router.get("/get-teams/{user_id}", response_model=List[TeamResponse])
async def get_teams(user_id: str):
    """
        This endpoint retrieves all teams for a user.
        
        Args:
            user_id (str): The user ID.
            
        Returns:
            List[TeamResponse]: List of teams.
    """
    try:
        teams = agent_manager.get_all_teams(user_id)
        return teams
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error"}
        )
        
@router.post("/team-run", response_model=TeamResponse)
async def team_run(request: Request):
    """
    This endpoint runs a team.
    
    Args:
        request (Request): The request body.
        
    Returns:
        TeamResponse: The team data.
    """
    try:
        body = await request.body()
        json_body = json.loads(body)
        request_body = RunTeamRequest(**json_body)

        team = agent_manager.get_team_by_id(
            team_id=request_body.team_id,
            session_id=request.state.user.get("sid"),
            user_id=request.state.user.get("sub")
        )
        response = team.run(
            message=request_body.message,
            stream=False
        )
        
        return JSONResponse(
            content={"response": response.content},
            status_code=200 
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error"}
        )