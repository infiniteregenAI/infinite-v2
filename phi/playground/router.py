import base64
from io import BytesIO
from typing import Any, List, Optional, AsyncGenerator, Dict, cast, Union, Generator
from uuid import uuid4
import os
import traceback
import json 

from utils.constants import AVAILABLE_TOOLS
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from schemas.agents_schema import AgentResponse, UserAgentsResponse
from dotenv import load_dotenv
from phi.agent.agent import Agent, RunResponse
from phi.agent.session import AgentSession
from phi.workflow.workflow import Workflow
from phi.workflow.session import WorkflowSession
from phi.playground.operator import (
    format_tools,
    get_agent_by_id,
    get_session_title,
    get_session_title_from_workflow_session,
    get_workflow_by_id,
)
from phi.tools.hackernews import HackerNews
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k
from phi.utils.log import logger
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from phi.playground.schemas import (
    AgentGetResponse,
    AgentSessionsRequest,
    AgentSessionsResponse,
    AgentRenameRequest,
    AgentModel,
    AgentSessionDeleteRequest,
    WorkflowRunRequest,
    WorkflowSessionsRequest,
    WorkflowRenameRequest,
)

load_dotenv()

agents_json_file_path="agents.json"
DB_URL = os.getenv("DB_URL")

def get_playground_router(
    agents: Optional[List[Agent]] = None, workflows: Optional[List[Workflow]] = None
) -> APIRouter:
    tool_map = {
            "HackerNews": HackerNews(),
            "DuckDuckGo": DuckDuckGo(),
            "Newspaper4k": Newspaper4k(),
    }
    playground_router = APIRouter(prefix="/playground", tags=["Playground"])
    if agents is None and workflows is None:
        raise ValueError("Either agents or workflows must be provided.")

    @playground_router.get("/status")
    def playground_status():
        return {"playground": "available"}
    
    @playground_router.get("/hello")
    def hello():
        return {"message": "Hello, World!"}
    
    @playground_router.post("/create/agent")
    def create_agent(
        name: str,
        user_id: Optional[str] = None,
        description: Optional[str] = None,
        role: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
    ):
        agent_id = "agent_"+str(uuid4())
        agent = Agent(
            name=name,
            role=role,
            agent_id=agent_id,
            tools=[tool_map[tool] for tool in tools] if tools else None,
            description=description,
            instructions=instructions,
            search_knowledge=True,
            knowledge_base=PDFUrlKnowledgeBase(
                urls=urls,
                vector_db=PgVector(table_name=f"{agent_id}_knowledge", db_url=DB_URL, search_type=SearchType.hybrid)
            ),
            show_tool_calls=True,
            markdown=True,
            add_datetime_to_instructions=True,
            user_id=user_id,
            storage=SqlAgentStorage(table_name=f"{agent_id}_ai_sessions", db_file="tmp/agents_sessions.db"),
            add_history_to_messages=True,
            num_history_responses=5,
        )
        agents.append(agent)
        
        return {"message": f"Agent {name} created successfully", "agent_id": agent_id}
        

    @playground_router.get("/agent/get", response_model=List[AgentGetResponse])
    def agent_get():
        agent_list: List[AgentGetResponse] = []
        if agents is None:
            return agent_list

        for agent in agents:
            agent_tools = agent.get_tools()
            formatted_tools = format_tools(agent_tools)

            name = agent.model.name or agent.model.__class__.__name__ if agent.model else None
            provider = agent.model.provider or agent.model.__class__.__name__ if agent.model else None
            model_id = agent.model.id if agent.model else None

            if provider and model_id:
                provider = f"{provider} {model_id}"
            elif name and model_id:
                provider = f"{name} {model_id}"
            elif model_id:
                provider = model_id
            else:
                provider = ""

            agent_list.append(
                AgentGetResponse(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    model=AgentModel(
                        name=name,
                        model=model_id,
                        provider=provider,
                    ),
                    add_context=agent.add_context,
                    tools=formatted_tools,
                    memory={"name": agent.memory.db.__class__.__name__} if agent.memory and agent.memory.db else None,
                    storage={"name": agent.storage.__class__.__name__} if agent.storage else None,
                    knowledge={"name": agent.knowledge.__class__.__name__} if agent.knowledge else None,
                    description=agent.description,
                    instructions=agent.instructions,
                )
            )

        return agent_list

    def chat_response_streamer(
        agent: Agent, message: str, images: Optional[List[Union[str, Dict]]] = None
    ) -> Generator:
        run_response = agent.run(message, images=images, stream=True, stream_intermediate_steps=True)
        for run_response_chunk in run_response:
            run_response_chunk = cast(RunResponse, run_response_chunk)
            yield run_response_chunk.to_json()

    def process_image(file: UploadFile) -> List[Union[str, Dict]]:
        content = file.file.read()
        encoded = base64.b64encode(content).decode("utf-8")

        image_info = {"filename": file.filename, "content_type": file.content_type, "size": len(content)}
        return [encoded, image_info]

    @playground_router.post("/agent/run")
    def agent_run(
        message: str = Form(...),
        agent_id: str = Form(...),
        stream: bool = Form(True),
        monitor: bool = Form(False),
        session_id: Optional[str] = Form(None),
        user_id: Optional[str] = Form(None),
        files: Optional[List[UploadFile]] = File(None),
        image: Optional[UploadFile] = File(None),
    ):
        logger.debug(f"AgentRunRequest: {message} {agent_id} {stream} {monitor} {session_id} {user_id} {files}")
        agent = get_agent_by_id(agent_id, agents)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        if files:
            if agent.knowledge is None:
                raise HTTPException(status_code=404, detail="KnowledgeBase not found")

        if session_id is not None:
            logger.debug(f"Continuing session: {session_id}")
        else:
            logger.debug("Creating new session")

        # Create a new instance of this agent
        new_agent_instance = agent.deep_copy(update={"session_id": session_id})
        if user_id is not None:
            new_agent_instance.user_id = user_id

        if monitor:
            new_agent_instance.monitoring = True
        else:
            new_agent_instance.monitoring = False

        base64_image: Optional[List[Union[str, Dict]]] = None
        if image:
            base64_image = process_image(image)

        if files:
            for file in files:
                if file.content_type == "application/pdf":
                    from phi.document.reader.pdf import PDFReader

                    contents = file.file.read()
                    pdf_file = BytesIO(contents)
                    pdf_file.name = file.filename
                    file_content = PDFReader().read(pdf_file)
                    if agent.knowledge is not None:
                        agent.knowledge.load_documents(file_content)
                elif file.content_type == "text/csv":
                    from phi.document.reader.csv_reader import CSVReader

                    contents = file.file.read()
                    csv_file = BytesIO(contents)
                    csv_file.name = file.filename
                    file_content = CSVReader().read(csv_file)
                    if agent.knowledge is not None:
                        agent.knowledge.load_documents(file_content)
                elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    from phi.document.reader.docx import DocxReader

                    contents = file.file.read()
                    docx_file = BytesIO(contents)
                    docx_file.name = file.filename
                    file_content = DocxReader().read(docx_file)
                    if agent.knowledge is not None:
                        agent.knowledge.load_documents(file_content)
                elif file.content_type == "text/plain":
                    from phi.document.reader.text import TextReader

                    contents = file.file.read()
                    text_file = BytesIO(contents)
                    text_file.name = file.filename
                    file_content = TextReader().read(text_file)
                    if agent.knowledge is not None:
                        agent.knowledge.load_documents(file_content)
                else:
                    raise HTTPException(status_code=400, detail="Unsupported file type")

        if stream:
            return StreamingResponse(
                chat_response_streamer(new_agent_instance, message, images=base64_image),
                media_type="text/event-stream",
            )
        else:
            run_response = cast(
                RunResponse,
                new_agent_instance.run(
                    message,
                    images=base64_image,
                    stream=False,
                ),
            )
            return run_response.model_dump_json()

    @playground_router.post("/agent/sessions/all")
    def get_agent_sessions(body: AgentSessionsRequest):
        logger.debug(f"AgentSessionsRequest: {body}")
        agent = get_agent_by_id(body.agent_id, agents)
        if agent is None:
            return JSONResponse(status_code=404, content="Agent not found.")

        if agent.storage is None:
            return JSONResponse(status_code=404, content="Agent does not have storage enabled.")

        agent_sessions: List[AgentSessionsResponse] = []
        all_agent_sessions: List[AgentSession] = agent.storage.get_all_sessions(user_id=body.user_id)
        for session in all_agent_sessions:
            title = get_session_title(session)
            agent_sessions.append(
                AgentSessionsResponse(
                    title=title,
                    session_id=session.session_id,
                    session_name=session.session_data.get("session_name") if session.session_data else None,
                    created_at=session.created_at,
                )
            )
        return agent_sessions

    @playground_router.post("/agent/sessions/{session_id}")
    def get_agent_session(session_id: str, body: AgentSessionsRequest):
        logger.debug(f"AgentSessionsRequest: {body}")
        agent = get_agent_by_id(body.agent_id, agents)
        if agent is None:
            return JSONResponse(status_code=404, content="Agent not found.")

        if agent.storage is None:
            return JSONResponse(status_code=404, content="Agent does not have storage enabled.")

        agent_session: Optional[AgentSession] = agent.storage.read(session_id)
        if agent_session is None:
            return JSONResponse(status_code=404, content="Session not found.")

        return agent_session

    @playground_router.post("/agent/session/rename")
    def agent_rename(body: AgentRenameRequest):
        agent = get_agent_by_id(body.agent_id, agents)
        if agent is None:
            return JSONResponse(status_code=404, content=f"couldn't find agent with {body.agent_id}")

        agent.session_id = body.session_id
        agent.rename_session(body.name)
        return JSONResponse(content={"message": f"successfully renamed agent {agent.name}"})

    @playground_router.post("/agent/session/delete")
    def agent_session_delete(body: AgentSessionDeleteRequest):
        agent = get_agent_by_id(body.agent_id, agents)
        if agent is None:
            return JSONResponse(status_code=404, content="Agent not found.")

        if agent.storage is None:
            return JSONResponse(status_code=404, content="Agent does not have storage enabled.")

        all_agent_sessions: List[AgentSession] = agent.storage.get_all_sessions(user_id=body.user_id)
        for session in all_agent_sessions:
            if session.session_id == body.session_id:
                agent.delete_session(body.session_id)
                return JSONResponse(content={"message": f"successfully deleted agent {agent.name}"})

        return JSONResponse(status_code=404, content="Session not found.")

    @playground_router.get("/workflows/get")
    def get_workflows():
        if workflows is None:
            return []

        return [
            {"id": workflow.workflow_id, "name": workflow.name, "description": workflow.description}
            for workflow in workflows
        ]

    @playground_router.get("/workflow/inputs/{workflow_id}")
    def get_workflow_inputs(workflow_id: str):
        workflow = get_workflow_by_id(workflow_id, workflows)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "parameters": workflow._run_parameters or {},
        }

    @playground_router.get("/workflow/config/{workflow_id}")
    def get_workflow_config(workflow_id: str):
        workflow = get_workflow_by_id(workflow_id, workflows)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return {
            "storage": workflow.storage.__class__.__name__ if workflow.storage else None,
        }

    @playground_router.post("/workflow/{workflow_id}/run")
    def run_workflow(workflow_id: str, body: WorkflowRunRequest):
        # Retrieve the workflow by ID
        workflow = get_workflow_by_id(workflow_id, workflows)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Create a new instance of this workflow
        new_workflow_instance = workflow.deep_copy(update={"workflow_id": workflow_id})
        new_workflow_instance.user_id = body.user_id

        # Return based on the response type
        try:
            if new_workflow_instance._run_return_type == "RunResponse":
                # Return as a normal response
                return new_workflow_instance.run(**body.input)
            else:
                # Return as a streaming response
                return StreamingResponse(
                    (result.model_dump_json() for result in new_workflow_instance.run(**body.input)),
                    media_type="text/event-stream",
                )
        except Exception as e:
            # Handle unexpected runtime errors
            raise HTTPException(status_code=500, detail=f"Error running workflow: {str(e)}")

    @playground_router.post("/workflow/{workflow_id}/session/all")
    def get_all_workflow_sessions(workflow_id: str, body: WorkflowSessionsRequest):
        # Retrieve the workflow by ID
        workflow = get_workflow_by_id(workflow_id, workflows)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Ensure storage is enabled for the workflow
        if not workflow.storage:
            raise HTTPException(status_code=404, detail="Workflow does not have storage enabled")

        # Retrieve all sessions for the given workflow and user
        try:
            all_workflow_sessions: List[WorkflowSession] = workflow.storage.get_all_sessions(
                user_id=body.user_id, workflow_id=workflow_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving sessions: {str(e)}")

        # Return the sessions
        return [
            {
                "title": get_session_title_from_workflow_session(session),
                "session_id": session.session_id,
                "session_name": session.session_data.get("session_name") if session.session_data else None,
                "created_at": session.created_at,
            }
            for session in all_workflow_sessions
        ]

    @playground_router.post("/workflow/{workflow_id}/session/{session_id}")
    def get_workflow_session(workflow_id: str, session_id: str, body: WorkflowSessionsRequest):
        # Retrieve the workflow by ID
        workflow = get_workflow_by_id(workflow_id, workflows)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Ensure storage is enabled for the workflow
        if not workflow.storage:
            raise HTTPException(status_code=404, detail="Workflow does not have storage enabled")

        # Retrieve the specific session
        try:
            workflow_session: Optional[WorkflowSession] = workflow.storage.read(session_id, body.user_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")

        if not workflow_session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Return the session
        return workflow_session

    @playground_router.post("/workflow/{workflow_id}/session/{session_id}/rename")
    def workflow_rename(workflow_id: str, session_id: str, body: WorkflowRenameRequest):
        workflow = get_workflow_by_id(workflow_id, workflows)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        workflow.rename_session(session_id, body.name)
        return JSONResponse(content={"message": f"successfully renamed workflow {workflow.name}"})

    @playground_router.post("/workflow/{workflow_id}/session/{session_id}/delete")
    def workflow_delete(workflow_id: str, session_id: str):
        workflow = get_workflow_by_id(workflow_id, workflows)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        workflow.delete_session(session_id)
        return JSONResponse(content={"message": f"successfully deleted workflow {workflow.name}"})

    return playground_router


def get_async_playground_router(
    agents: Optional[List[Agent]] = None, workflows: Optional[List[Workflow]] = None
) -> APIRouter:
    tool_map = {
            "HackerNews": HackerNews(),
            "DuckDuckGo": DuckDuckGo(),
            "Newspaper4k": Newspaper4k(),
    }
    playground_router = APIRouter(prefix="/playground", tags=["Playground"])
    if agents is None and workflows is None:
        raise ValueError("Either agents or workflows must be provided.")

    @playground_router.get("/status")
    async def playground_status():
        return {"playground": "available"}
    
    @playground_router.get("/hello")
    async def hello():
        return {"message": "Hello, World!"}
    
    @playground_router.post("/create/agent")
    async def create_agent(
        name: str,
        user_id: Optional[str] = None,
        description: Optional[str] = None,
        role: Optional[str] = None,
        instructions: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
    ):
        agent_id = "agent_"+str(uuid4())
        agent = Agent(
            name=name,
            role=role,
            agent_id=agent_id,
            tools=[tool_map[tool] for tool in tools] if tools else None,
            description=description,
            instructions=instructions,
            search_knowledge=True,
            knowledge_base=PDFUrlKnowledgeBase(
                urls=urls,
                vector_db=PgVector(table_name=f"{agent_id}_knowledge", db_url=DB_URL, search_type=SearchType.hybrid)
            ),
            show_tool_calls=True,
            markdown=True,
            add_datetime_to_instructions=True,
            user_id=user_id,
            storage=SqlAgentStorage(table_name=f"{agent_id}_ai_sessions", db_file="tmp/agents_sessions.db"),
            add_history_to_messages=True,
            num_history_responses=5,
        )
        agent_data = AgentResponse(
            id=agent_id,
            name=name,
            role=role,
            tools=tools,
            description=description,
            instructions=instructions,
            urls=urls,
            markdown=True,
            show_tool_calls=True,
            add_datetime_to_instructions=True,
            user_id=user_id 
        ) 
        agents.append(agent)
        
        with open(agents_json_file_path, 'r') as f:
            json_agents = json.load(f)
        json_agents.append(agent_data.model_dump())    
        with open(agents_json_file_path, 'w') as f:
            json.dump(json_agents, f, indent=2)
        
        return {"message": f"Agent {name} created successfully", "agent_id": agent_id}
    
    @playground_router.put("/update/agent/{agent_id}")
    async def update_agent(
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        role: Optional[str] = None,
        instructions: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
    ):
        # Load existing agents
        with open(agents_json_file_path, 'r') as f:
            json_agents = json.load(f)

        # Find and update agent
        for agent in json_agents:
            if agent["id"] == agent_id:
                if name: agent["name"] = name
                if description: agent["description"] = description
                if role: agent["role"] = role
                if instructions: agent["instructions"] = instructions
                if tools: agent["tools"] = tools
                if urls: agent["urls"] = urls
                break
        else:
            return {"message": "Agent not found"}

        # Save updated agents list
        with open(agents_json_file_path, 'w') as f:
            json.dump(json_agents, f, indent=2)

        return {"message": f"Agent {agent_id} updated successfully"}
    
    @playground_router.get("/available-tools")
    async def get_available_tools():
        """
            This endpoint returns the available tools.
            
            Returns:
                JSONResponse: The response body.
        """
        try:
            return JSONResponse(
                status_code=200,
                content={"available_tools": AVAILABLE_TOOLS}
            )
        except Exception as e:
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"message": "Internal server error"}
            )


    @playground_router.delete("/delete/agent/{agent_id}")
    async def delete_agent(agent_id: str):
        # Load existing agents
        with open(agents_json_file_path, 'r') as f:
            json_agents = json.load(f)

        # Remove the agent from the list
        json_agents = [agent for agent in json_agents if agent["id"] != agent_id]

        # Save updated agents list
        with open(agents_json_file_path, 'w') as f:
            json.dump(json_agents, f, indent=2)

        return {"message": f"Agent {agent_id} deleted successfully"}

    @playground_router.get("/agents/reserved/", response_model=UserAgentsResponse)
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
    
    @playground_router.get("/agents/user/{user_id}", response_model=UserAgentsResponse)
    async def get_agents_by_user(user_id: str):
        """
        Get all agents created by a specific user.
        
        Args:
            user_id (str): The ID of the user whose agents to retrieve.
            
        Returns:
            UserAgentsResponse: List of agents belonging to the user.
        """
        try:
            with open(agents_json_file_path, 'r') as f:
                json_agents = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"message": "No agents found"}

        user_agents = [agent for agent in json_agents if agent.get("user_id") == user_id]

        return UserAgentsResponse(agents=user_agents)


    @playground_router.get("/agent/get", response_model=List[AgentGetResponse])
    async def agent_get():
        agent_list: List[AgentGetResponse] = []
        if agents is None:
            return agent_list

        for agent in agents:
            agent_tools = agent.get_tools()
            formatted_tools = format_tools(agent_tools)

            name = agent.model.name or agent.model.__class__.__name__ if agent.model else None
            provider = agent.model.provider or agent.model.__class__.__name__ if agent.model else None
            model_id = agent.model.id if agent.model else None

            if provider and model_id:
                provider = f"{provider} {model_id}"
            elif name and model_id:
                provider = f"{name} {model_id}"
            elif model_id:
                provider = model_id
            else:
                provider = ""

            agent_list.append(
                AgentGetResponse(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    model=AgentModel(
                        name=name,
                        model=model_id,
                        provider=provider,
                    ),
                    add_context=agent.add_context,
                    tools=formatted_tools,
                    memory={"name": agent.memory.db.__class__.__name__} if agent.memory and agent.memory.db else None,
                    storage={"name": agent.storage.__class__.__name__} if agent.storage else None,
                    knowledge={"name": agent.knowledge.__class__.__name__} if agent.knowledge else None,
                    description=agent.description,
                    instructions=agent.instructions,
                )
            )

        return agent_list

    async def chat_response_streamer(
        agent: Agent,
        message: str,
        images: Optional[List[Union[str, Dict]]] = None,
        audio_file_content: Optional[Any] = None,
        video_file_content: Optional[Any] = None,
    ) -> AsyncGenerator:
        run_response = await agent.arun(
            message,
            images=images,
            audio=audio_file_content,
            videos=video_file_content,
            stream=True,
            stream_intermediate_steps=True,
        )
        async for run_response_chunk in run_response:
            run_response_chunk = cast(RunResponse, run_response_chunk)
            yield run_response_chunk.to_json()

    async def process_image(file: UploadFile) -> List[Union[str, Dict]]:
        content = file.file.read()
        encoded = base64.b64encode(content).decode("utf-8")

        image_info = {"filename": file.filename, "content_type": file.content_type, "size": len(content)}

        return [encoded, image_info]

    @playground_router.post("/agent/run")
    async def agent_run(
        message: str = Form(...),
        agent_id: str = Form(...),
        stream: bool = Form(True),
        monitor: bool = Form(False),
        session_id: Optional[str] = Form(None),
        user_id: Optional[str] = Form(None),
        files: Optional[List[UploadFile]] = File(None),
        image: Optional[UploadFile] = File(None),
    ):
        logger.debug(f"AgentRunRequest: {message} {session_id} {user_id} {agent_id}")
        agent = get_agent_by_id(agent_id, agents)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        if files:
            if agent.knowledge is None:
                raise HTTPException(status_code=404, detail="KnowledgeBase not found")

        if session_id is not None:
            logger.debug(f"Continuing session: {session_id}")
        else:
            logger.debug("Creating new session")

        # Create a new instance of this agent
        new_agent_instance = agent.deep_copy(update={"session_id": session_id})
        if user_id is not None:
            new_agent_instance.user_id = user_id

        if monitor:
            new_agent_instance.monitoring = True
        else:
            new_agent_instance.monitoring = False

        base64_image: Optional[List[Union[str, Dict]]] = None
        if image:
            base64_image = await process_image(image)

        if files:
            for file in files:
                if file.content_type == "application/pdf":
                    from phi.document.reader.pdf import PDFReader

                    contents = await file.read()
                    pdf_file = BytesIO(contents)
                    pdf_file.name = file.filename
                    file_content = PDFReader().read(pdf_file)
                    if agent.knowledge is not None:
                        agent.knowledge.load_documents(file_content)
                elif file.content_type == "text/csv":
                    from phi.document.reader.csv_reader import CSVReader

                    contents = await file.read()
                    csv_file = BytesIO(contents)
                    csv_file.name = file.filename
                    file_content = CSVReader().read(csv_file)
                    if agent.knowledge is not None:
                        agent.knowledge.load_documents(file_content)
                elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    from phi.document.reader.docx import DocxReader

                    contents = await file.read()
                    docx_file = BytesIO(contents)
                    docx_file.name = file.filename
                    file_content = DocxReader().read(docx_file)
                    if agent.knowledge is not None:
                        agent.knowledge.load_documents(file_content)
                elif file.content_type == "text/plain":
                    from phi.document.reader.text import TextReader

                    contents = await file.read()
                    text_file = BytesIO(contents)
                    text_file.name = file.filename
                    file_content = TextReader().read(text_file)
                    if agent.knowledge is not None:
                        agent.knowledge.load_documents(file_content)
                else:
                    raise HTTPException(status_code=400, detail="Unsupported file type")

        if stream:
            return StreamingResponse(
                chat_response_streamer(new_agent_instance, message, images=base64_image),
                media_type="text/event-stream",
            )
        else:
            run_response = cast(
                RunResponse,
                await new_agent_instance.arun(
                    message,
                    images=base64_image,
                    stream=False,
                ),
            )
            return run_response.model_dump_json()

    @playground_router.post("/agent/sessions/all")
    async def get_agent_sessions(body: AgentSessionsRequest):
        logger.debug(f"AgentSessionsRequest: {body}")
        agent = get_agent_by_id(body.agent_id, agents)
        if agent is None:
            return JSONResponse(status_code=404, content="Agent not found.")

        if agent.storage is None:
            return JSONResponse(status_code=404, content="Agent does not have storage enabled.")

        agent_sessions: List[AgentSessionsResponse] = []
        all_agent_sessions: List[AgentSession] = agent.storage.get_all_sessions(user_id=body.user_id)
        for session in all_agent_sessions:
            title = get_session_title(session)
            agent_sessions.append(
                AgentSessionsResponse(
                    title=title,
                    session_id=session.session_id,
                    session_name=session.session_data.get("session_name") if session.session_data else None,
                    created_at=session.created_at,
                )
            )
        return agent_sessions

    @playground_router.post("/agent/sessions/{session_id}")
    async def get_agent_session(session_id: str, body: AgentSessionsRequest):
        logger.debug(f"AgentSessionsRequest: {body}")
        agent = get_agent_by_id(body.agent_id, agents)
        if agent is None:
            return JSONResponse(status_code=404, content="Agent not found.")

        if agent.storage is None:
            return JSONResponse(status_code=404, content="Agent does not have storage enabled.")

        agent_session: Optional[AgentSession] = agent.storage.read(session_id, body.user_id)
        if agent_session is None:
            return JSONResponse(status_code=404, content="Session not found.")

        return agent_session

    @playground_router.post("/agent/session/rename")
    async def agent_rename(body: AgentRenameRequest):
        agent = get_agent_by_id(body.agent_id, agents)
        if agent is None:
            return JSONResponse(status_code=404, content=f"couldn't find agent with {body.agent_id}")

        agent.session_id = body.session_id
        agent.rename_session(body.name)
        return JSONResponse(content={"message": f"successfully renamed agent {agent.name}"})

    @playground_router.post("/agent/session/delete")
    async def agent_session_delete(body: AgentSessionDeleteRequest):
        agent = get_agent_by_id(body.agent_id, agents)
        if agent is None:
            return JSONResponse(status_code=404, content="Agent not found.")

        if agent.storage is None:
            return JSONResponse(status_code=404, content="Agent does not have storage enabled.")

        all_agent_sessions: List[AgentSession] = agent.storage.get_all_sessions(user_id=body.user_id)
        for session in all_agent_sessions:
            if session.session_id == body.session_id:
                agent.delete_session(body.session_id)
                return JSONResponse(content={"message": f"successfully deleted agent {agent.name}"})

        return JSONResponse(status_code=404, content="Session not found.")

    @playground_router.get("/workflows/get")
    async def get_workflows():
        if workflows is None:
            return []

        return [
            {"id": workflow.workflow_id, "name": workflow.name, "description": workflow.description}
            for workflow in workflows
        ]

    @playground_router.get("/workflow/inputs/{workflow_id}")
    async def get_workflow_inputs(workflow_id: str):
        workflow = get_workflow_by_id(workflow_id, workflows)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "parameters": workflow._run_parameters or {},
        }

    @playground_router.get("/workflow/config/{workflow_id}")
    async def get_workflow_config(workflow_id: str):
        workflow = get_workflow_by_id(workflow_id, workflows)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return {
            "storage": workflow.storage.__class__.__name__ if workflow.storage else None,
        }

    @playground_router.post("/workflow/{workflow_id}/run")
    async def run_workflow(workflow_id: str, body: WorkflowRunRequest):
        # Retrieve the workflow by ID
        workflow = get_workflow_by_id(workflow_id, workflows)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        if body.session_id is not None:
            logger.debug(f"Continuing session: {body.session_id}")
        else:
            logger.debug("Creating new session")

        # Create a new instance of this workflow
        new_workflow_instance = workflow.deep_copy(update={"workflow_id": workflow_id, "session_id": body.session_id})
        new_workflow_instance.user_id = body.user_id

        # Return based on the response type
        try:
            if new_workflow_instance._run_return_type == "RunResponse":
                # Return as a normal response
                return new_workflow_instance.run(**body.input)
            else:
                # Return as a streaming response
                return StreamingResponse(
                    (result.model_dump_json() for result in new_workflow_instance.run(**body.input)),
                    media_type="text/event-stream",
                )
        except Exception as e:
            # Handle unexpected runtime errors
            raise HTTPException(status_code=500, detail=f"Error running workflow: {str(e)}")

    @playground_router.post("/workflow/{workflow_id}/session/all")
    async def get_all_workflow_sessions(workflow_id: str, body: WorkflowSessionsRequest):
        # Retrieve the workflow by ID
        workflow = get_workflow_by_id(workflow_id, workflows)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Ensure storage is enabled for the workflow
        if not workflow.storage:
            raise HTTPException(status_code=404, detail="Workflow does not have storage enabled")

        # Retrieve all sessions for the given workflow and user
        try:
            all_workflow_sessions: List[WorkflowSession] = workflow.storage.get_all_sessions(
                user_id=body.user_id, workflow_id=workflow_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving sessions: {str(e)}")

        # Return the sessions
        return [
            {
                "title": get_session_title_from_workflow_session(session),
                "session_id": session.session_id,
                "session_name": session.session_data.get("session_name") if session.session_data else None,
                "created_at": session.created_at,
            }
            for session in all_workflow_sessions
        ]

    @playground_router.post("/workflow/{workflow_id}/session/{session_id}")
    async def get_workflow_session(workflow_id: str, session_id: str, body: WorkflowSessionsRequest):
        # Retrieve the workflow by ID
        workflow = get_workflow_by_id(workflow_id, workflows)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Ensure storage is enabled for the workflow
        if not workflow.storage:
            raise HTTPException(status_code=404, detail="Workflow does not have storage enabled")

        # Retrieve the specific session
        try:
            workflow_session: Optional[WorkflowSession] = workflow.storage.read(session_id, body.user_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")

        if not workflow_session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Return the session
        return workflow_session

    @playground_router.post("/workflow/{workflow_id}/session/{session_id}/rename")
    async def workflow_rename(workflow_id: str, session_id: str, body: WorkflowRenameRequest):
        workflow = get_workflow_by_id(workflow_id, workflows)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        workflow.rename_session(session_id, body.name)
        return JSONResponse(content={"message": f"successfully renamed workflow {workflow.name}"})

    @playground_router.post("/workflow/{workflow_id}/session/{session_id}/delete")
    async def workflow_delete(workflow_id: str, session_id: str):
        workflow = get_workflow_by_id(workflow_id, workflows)
        if workflow is None:
            raise HTTPException(status_code=404, detail="Workflow not found")

        workflow.delete_session(session_id)
        return JSONResponse(content={"message": f"successfully deleted workflow {workflow.name}"})

    return playground_router
