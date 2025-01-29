import traceback
import logging
import json
import sqlite3
import asyncio
import re

from fastapi import APIRouter,Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from schemas.agents_schema import CreateAgentRequest, CreateAgentResponse, AgentResponse, UserAgentsResponse , RunAgentRequest
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
    
@router.put("/update-agent/{agent_name}", response_model=CreateAgentResponse)
async def update_agent(agent_name: str, request: CreateAgentRequest):
    """
        This endpoint updates an existing agent.
        
        Args:
            agent_name (str): The name of the agent to update.
            request (CreateAgentRequest): The request body.
            
        Returns:
            CreateAgentResponse: The response body.
    """
    invalid_tools = [tool for tool in request.tools if tool not in AVAILABLE_TOOLS]
    
    if invalid_tools:
        logger.error(f"Invalid tools selected: {', '.join(invalid_tools)}. Available tools are: {', '.join(AVAILABLE_TOOLS)}.")
        return JSONResponse(
            status_code=400,
            content={
                "message": f"Invalid tools selected: {', '.join(invalid_tools)}. Available tools are: {', '.join(AVAILABLE_TOOLS)}."
            }
        )
    try:
        agent = agent_manager.update_agent(
            name=agent_name,
            role=request.role,
            tools=request.tools,
            description=request.description,
            instructions=request.instructions,
            urls=request.urls,  
        )
        
        return CreateAgentResponse(
            message="Agent updated successfully",
            agent=AgentResponse(
                id=agent.id,
                name=agent.name,
                role=agent.role,
                tools=request.tools,
                description=request.description,
                instructions=request.instructions,
                urls=request.urls,  
                markdown=True,
                show_tool_calls=True,
                add_datetime_to_instructions=True
            )
        )
        
    except ValueError as e:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=400, content={"message": str(e)})
    
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": str(e)})

@router.post("/run")
async def run_agent(request: Request):
    """
    This endpoint runs an agent and streams the response.

    Args:
        request (Request): The request body.

    Returns:
        StreamingResponse: The streamed response.
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
        cleaned_content = re.sub(r"Running:\n - .*?\n\n", "", response.content, flags=re.DOTALL).strip()

        async def stream_content():
            chunk_size = 1024 
            for i in range(0, len(cleaned_content), chunk_size):
                yield cleaned_content[i:i + chunk_size]
                await asyncio.sleep(0.1)  

        return StreamingResponse(stream_content(), media_type="text/plain")
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
        
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": str(e)})