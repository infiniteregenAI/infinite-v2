import traceback
import logging
import json
import sqlite3

from fastapi import APIRouter,Request
from fastapi.responses import JSONResponse

from schemas.agents_schema import CreateAgentRequest, CreateAgentResponse, AgentResponse, UserAgentsResponse , RunAgentRequest, UpdateAgentRequest, UpdateAgentResponse, DeleteAgentResponse
from utils.constants import AVAILABLE_TOOLS
from utils.agent_manager import AgentManager

agent_manager = AgentManager()

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/create-agent", response_model=CreateAgentResponse)
async def create_agent(request: Request):
    """
        This endpoint creates a new agent.
        
        Args:
            request (Request): The request body.
            
        Returns:
            CreateAgentResponse: The response body.
    """
    
    body = await request.body()
    json_body = json.loads(body)
    request_body = CreateAgentRequest(**json_body)
    
    invalid_tools = [tool for tool in request_body.tools if tool not in AVAILABLE_TOOLS]
    if invalid_tools:
        return JSONResponse(
            status_code=400,
            content={
                "message": f"Invalid tools selected: {', '.join(invalid_tools)}. Available tools are: {', '.join(AVAILABLE_TOOLS)}."
            }
        )
    try:
        agent = agent_manager.create_agent(
            user_id=request.state.user.get("sub"),
            name=request_body.name,
            role=request_body.role,
            tools=request_body.tools,
            description=request_body.description,
            instructions=request_body.instructions,
            urls=request_body.urls,  
        )
        
        return CreateAgentResponse(
            message="Agent created successfully",
            agent=AgentResponse(
                id=agent.id,
                user_id=agent.user_id,
                name=agent.name,
                role=agent.role,
                tools=request_body.tools,
                description=request_body.description,
                instructions=request_body.instructions,
                urls=request_body.urls,  
                markdown=True,
                show_tool_calls=True,
                add_datetime_to_instructions=True
            )
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": str(e)})
    
@router.put("/update-agent/", response_model=UpdateAgentResponse)
async def update_agent(request: Request):
    """
    Update specific fields of an existing agent.
    Only role, tools, description, instructions, and urls can be updated.
    
    Returns:
        UpdateAgentResponse: Contains success message and updated agent data
    """
    try:
        body = await request.body()
        json_body = json.loads(body)
        request_body = UpdateAgentRequest(**json_body)
        
        # Validate tools
        invalid_tools = [tool for tool in request_body.tools if tool not in AVAILABLE_TOOLS]
        if invalid_tools:
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Invalid tools selected: {', '.join(invalid_tools)}. Available tools are: {', '.join(AVAILABLE_TOOLS)}."
                }
            )
            
        # Update agent
        updated_agent = agent_manager.update_agent(
            user_id=request.state.user["sub"],
            agent_id=request_body.agent_id,
            role=request_body.role,
            tools=request_body.tools,
            description=request_body.description,
            instructions=request_body.instructions,
            urls=request_body.urls
        )
        
        # Return response with message and updated agent data
        return UpdateAgentResponse(
            message="Agent updated successfully",
            agent=updated_agent
        )
    except ValueError as e:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=400, content={"message": str(e)})
        
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": str(e)})

@router.post("/run")
async def run_agent(request : Request):
    """
        This endpoint runs an agent.
        
        Args:
            request (Request): The request body.
            
        Returns:
            StreamResponse: The response body.
    """
    try:
        
        body = await request.body()
        json_body = json.loads(body)
        request_body = RunAgentRequest(**json_body)
        
        
        agent = agent_manager.get_agent_by_id(
            agent_id=request_body.agent_id,
            session_id=request.state.user.get("sid"),
            user_id=request.state.user.get("sub")
        )
        response = agent.run(
            message=request_body.message,
            stream=False
        )
        
        return JSONResponse(content={"response": response.content} , status_code=200)
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": str(e)})

@router.get("/agents/user/{user_id}", response_model=UserAgentsResponse)
async def get_agents_by_user(user_id: str):
    """
    Get all agents created by a specific user.
    
    Args:
        user_id (str): The ID of the user whose agents to retrieve.
        
    Returns:
        UserAgentsResponse: List of agents belonging to the user.
    """
    try:
        agents = agent_manager.get_agents_by_user(user_id)
        return UserAgentsResponse(agents=agents)
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": str(e)})

@router.get("/agents/reserved/", response_model=UserAgentsResponse)
async def get_reserved_agents():
    """
    Get all reserved agents.
    
    Returns:
        UserAgentsResponse: List of reserved agents.
    """
    try:
        # Load the agents from the JSON file
        with open("agents.json", "r") as file:
            agents = json.load(file)
        
        # Filter agents with IDs starting with "reservered_agent_"
        reserved_agents = [agent for agent in agents if agent["id"].startswith("reservered_agent_")]
        
        return UserAgentsResponse(agents=reserved_agents)
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": str(e)})
<<<<<<< Updated upstream
    
@router.get("/get-all-sessions/{agent_id}")
async def get_all_sessions(agent_id: str):
    try:
        conn = sqlite3.connect("D:/ai_swarm_backend/corev2/tmp/agents_sessions.db")
        conn.row_factory = sqlite3.Row  
        cursor = conn.cursor()

        # Quote the table name to handle special characters
        table_name = f'"{agent_id}_ai_sessions"'
        query = f"SELECT * FROM {table_name}"
        cursor.execute(query)
        records = cursor.fetchall()

        # Convert records to a list of dictionaries
        result = [dict(row) for row in records]

        # Close the connection
        conn.close()
        sessions = []
        for row in result:
            session = {
                "session_id": row["session_id"],
                "user_id": row["user_id"],
                "agent_id": row["agent_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                
            }
            sessions.append(session)
=======

@router.delete("/delete-agent/{agent_id}", response_model=DeleteAgentResponse)
async def delete_agent(agent_id: str, request: Request):
    """
    Delete an agent by its ID.
    
    Args:
        agent_id (str): The ID of the agent to delete
        request (Request): The request object containing user information
        
    Returns:
        DeleteAgentResponse: Contains success message and deleted agent data
    """
    try:
        # Delete the agent
        deleted_agent = agent_manager.delete_agent(
            user_id=request.state.user["sub"],
            agent_id=agent_id
        )
        
        # Return response with message and deleted agent data
        return DeleteAgentResponse(
            message="Agent deleted successfully",
            body=deleted_agent
        )
        
    except ValueError as e:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=400, content={"message": str(e)})
>>>>>>> Stashed changes
        
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": str(e)})