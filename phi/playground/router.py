import base64
from io import BytesIO
from typing import Any, List, Optional, AsyncGenerator, Dict, cast, Union, Generator
from uuid import uuid4
import os
import traceback
import json 
from typing import Annotated

from utils.constants import AVAILABLE_TOOLS
from fastapi import APIRouter, File, Form, HTTPException, UploadFile , Request, Depends, Query
from fastapi.responses import StreamingResponse, JSONResponse
from schemas.agents_schema import AgentResponse, UserAgentsResponse, UpdateAgentRequest, UpdateAgentResponse, DeleteAgentResponse
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
from phi.knowledge.website import WebsiteKnowledgeBase
from phi.knowledge.combined import CombinedKnowledgeBase
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
    AgentCreateRequest
)
from schemas.database import get_db, DatabaseOperations, get_db_session
from sqlalchemy.orm import Session
from sqlalchemy import text


load_dotenv()

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
        request: Request,
        body: AgentCreateRequest,
        db: Session = Depends(get_db)
    ):
        agent_id = "agent_"+str(uuid4())
        user_id = request.state.user.get("sub")
        
        pdf_knowledge_base = PDFUrlKnowledgeBase(
            urls=body.pdf_urls,
            vector_db=PgVector(
                table_name=f"{agent_id}_pdf_knowledge",
                db_url=DB_URL,
                search_type=SearchType.hybrid
            )
        )

        website_knowledge_base = WebsiteKnowledgeBase(
            urls=body.website_urls,
            max_links=10,
            vector_db=PgVector(
                table_name=f"{agent_id}_website_knowledge",
                db_url=DB_URL,
                search_type=SearchType.hybrid
            )
        )

        combined_knowledge_base = CombinedKnowledgeBase(
            sources=[pdf_knowledge_base, website_knowledge_base],
            vector_db=PgVector(
                table_name=f"{agent_id}_combined_knowledge",
                db_url=DB_URL,
                search_type=SearchType.hybrid
            )
        )

        agent = Agent(
            name=body.name,
            role=body.role,
            agent_id=agent_id,
            tools=[tool_map[tool] for tool in body.tools] if body.tools else None,
            description=body.description,
            instructions=body.instructions,
            search_knowledge=True,
            knowledge_base=combined_knowledge_base,
            show_tool_calls=True,
            markdown=True,
            add_datetime_to_instructions=True,
            user_id=user_id,
            storage=SqlAgentStorage(
                table_name=f"{agent_id}_ai_sessions",
                db_file="tmp/agents_sessions.db"
            ),
            add_history_to_messages=True,
            num_history_responses=5,
        )
        
        agents.append(agent)
        
        agent_data = {
            "id": agent_id,
            "name": body.name,
            "role": body.role,
            "tools": body.tools,
            "description": body.description,
            "instructions": body.instructions,  
            "pdf_urls": body.pdf_urls,
            "website_urls": body.website_urls,
            "markdown": True,
            "show_tool_calls": True,
            "add_datetime_to_instructions": True,
            "user_id": user_id
        }
        
        
        try:
            db_agent = DatabaseOperations.create_agent(db, agent_data)
            return {"status": "success", "agent_id": agent_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")
    
    @playground_router.put("/update/agent/{agent_id}")
    async def update_agent(
        request: Request,
        agent_id: str,
        body: UpdateAgentRequest,
        db: Session = Depends(get_db)
    ):
        user_id = request.state.user.get("sub")
        
        existing_agent = DatabaseOperations.get_agent(db, agent_id)
        if existing_agent.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to update this agent")
        
        instructions = body.instructions if body.instructions is not None else existing_agent.instructions
        
        if isinstance(instructions, str):
            instructions = [instructions]
        
        update_data = {
            "name": body.name if body.name is not None else existing_agent.name,
            "role": body.role if body.role is not None else existing_agent.role,
            "tools": body.tools if body.tools is not None else existing_agent.tools,
            "description": body.description if body.description is not None else existing_agent.description,
            "instructions": instructions,  
            "pdf_urls": body.pdf_urls if body.pdf_urls is not None else existing_agent.pdf_urls,
            "website_urls": body.website_urls if body.website_urls is not None else existing_agent.website_urls
        }
        
        try:
            updated_agent = DatabaseOperations.update_agent(db, agent_id, update_data)
            
            pdf_knowledge_base = PDFUrlKnowledgeBase(
                urls=update_data["pdf_urls"],
                vector_db=PgVector(
                    table_name=f"{agent_id}_pdf_knowledge",
                    db_url=DB_URL,
                    search_type=SearchType.hybrid
                )
            )
            
            website_knowledge_base = WebsiteKnowledgeBase(
                urls=update_data["website_urls"],
                max_links=10,
                vector_db=PgVector(
                    table_name=f"{agent_id}_website_knowledge",
                    db_url=DB_URL,
                    search_type=SearchType.hybrid
                )
            )
            
            combined_knowledge_base = CombinedKnowledgeBase(
                sources=[pdf_knowledge_base, website_knowledge_base],
                vector_db=PgVector(
                    table_name=f"{agent_id}_combined_knowledge",
                    db_url=DB_URL,
                    search_type=SearchType.hybrid
                )
            )
            
            for agent in agents:
                if agent.agent_id == agent_id:
                    agent.name = update_data["name"]
                    agent.role = update_data["role"]
                    agent.tools = [tool_map[tool] for tool in update_data["tools"]] if update_data["tools"] else None
                    agent.description = update_data["description"]
                    agent.instructions = instructions
                    agent.knowledge_base = combined_knowledge_base
                    break
            
            response = UpdateAgentResponse(
                message=f"Agent {agent_id} updated successfully",
                agent=AgentResponse(
                    id=updated_agent.id,
                    user_id=updated_agent.user_id,
                    name=updated_agent.name,
                    role=updated_agent.role,
                    tools=updated_agent.tools,
                    description=updated_agent.description,
                    instructions=instructions, 
                    pdf_urls=updated_agent.pdf_urls,
                    website_urls=updated_agent.website_urls,
                    markdown=updated_agent.markdown,
                    show_tool_calls=updated_agent.show_tool_calls,
                    add_datetime_to_instructions=updated_agent.add_datetime_to_instructions
                )
            )
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating agent: {str(e)}")
    
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
    async def delete_agent(
        request: Request,
        agent_id: str,
        db: Session = Depends(get_db)
    ):
        user_id = request.state.user.get("sub")
        
        try:
            existing_agent = DatabaseOperations.get_agent(db, agent_id)
            if existing_agent.user_id != user_id:
                raise HTTPException(
                    status_code=403, 
                    detail="You do not have permission to delete this agent. Only the agent's creator can delete it."
                )
            
            deleted_agent = DatabaseOperations.delete_agent(db, agent_id)
            
            try:
                with get_db_session() as session:
                    session.execute(text(f'DROP TABLE IF EXISTS "{agent_id}_pdf_knowledge"'))
                    session.execute(text(f'DROP TABLE IF EXISTS "{agent_id}_website_knowledge"'))
                    session.execute(text(f'DROP TABLE IF EXISTS "{agent_id}_combined_knowledge"'))
                    session.execute(text(f'DROP TABLE IF EXISTS "{agent_id}_ai_sessions"'))
            except Exception as e:
                print(f"Warning: Failed to clean up associated tables: {str(e)}")
            
            response = DeleteAgentResponse(
                message=f"Agent {agent_id} deleted successfully",
                body=AgentResponse(
                    id=existing_agent.id,
                    user_id=existing_agent.user_id,
                    name=existing_agent.name,
                    role=existing_agent.role,
                    tools=existing_agent.tools,
                    description=existing_agent.description,
                    instructions=existing_agent.instructions,
                    pdf_urls=existing_agent.pdf_urls,
                    website_urls=existing_agent.website_urls,
                    markdown=existing_agent.markdown,
                    show_tool_calls=existing_agent.show_tool_calls,
                    add_datetime_to_instructions=existing_agent.add_datetime_to_instructions
                )
            )
            
            return response
            
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @playground_router.get("/agents/", response_model=UserAgentsResponse)
    async def get_agents(
        request: Request,
        type: str = Query(None, description="Filter type: 'reserved' for reserved agents, 'user' for user agents, or None for both"),
        db: Session = Depends(get_db)
    ):
        try:
            user_id = request.state.user.get("sub")  

            if not user_id:
                raise HTTPException(status_code=401, detail="Authentication required")

            user_agents = []
            reserved_agents = []

            if type in [None, "user"]:
                db_agents = DatabaseOperations.get_agents_by_user(db, user_id)

                user_agents = [
                    AgentResponse(
                        id=agent.id,
                        user_id=agent.user_id,
                        name=agent.name,
                        role=agent.role,
                        tools=agent.tools,
                        description=agent.description,
                        instructions=agent.instructions if agent.instructions else [],  
                        pdf_urls=agent.pdf_urls,
                        website_urls=agent.website_urls,
                        markdown=agent.markdown,
                        show_tool_calls=agent.show_tool_calls,
                        add_datetime_to_instructions=agent.add_datetime_to_instructions
                    ) 
                    for agent in db_agents
                ]

            if type in [None, "reserved"]:
                try:
                    with open("agents.json", "r") as file:
                        agents_data = json.load(file)

                    reserved_agents = [
                        AgentResponse(
                            id=agent["id"],
                            user_id="reserved",
                            name=agent["name"],
                            role=agent["role"],
                            tools=agent["tools"],
                            description=agent["description"],
                            instructions=agent.get("instructions", []),  
                            pdf_urls=agent.get("pdf_urls", []),
                            website_urls=agent.get("website_urls", []),
                            markdown=agent.get("markdown", False),
                            show_tool_calls=agent.get("show_tool_calls", False),
                            add_datetime_to_instructions=agent.get("add_datetime_to_instructions", False)
                        )
                        for agent in agents_data if agent["id"].startswith("reserved_agent_")
                    ]
                except Exception as e:
                    logger.error(f"Error loading reserved agents: {e}")
                    reserved_agents = []

            if type == "user":
                agents = user_agents
            elif type == "reserved":
                agents = reserved_agents
            else:
                agents = user_agents + reserved_agents

            return UserAgentsResponse(agents=agents)

        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(traceback.format_exc())
            return JSONResponse(status_code=500, content={"message": str(e)})


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
        request: Request,
        message: str = Form(...),
        agent_id: str = Form(...),
        stream: bool = Form(True),
        monitor: bool = Form(False),
        session_id: Optional[str] = Form(None),
        files: Optional[List[UploadFile]] = File(None),
        image: Optional[UploadFile] = File(None),
    ):
        try:
            user_id = request.state.user.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="Authentication required")

            logger.debug(f"AgentRunRequest: {message} {session_id} {user_id} {agent_id}")

            agent = get_agent_by_id(agent_id, agents)
            if agent is None:
                raise HTTPException(status_code=404, detail="Agent not found")

            if files and agent.knowledge is None:
                raise HTTPException(status_code=404, detail="KnowledgeBase not found")

            logger.debug(f"User {user_id} is running agent {agent_id}")

            new_agent_instance = agent.deep_copy(update={"session_id": session_id, "user_id": user_id})
            new_agent_instance.monitoring = monitor

            base64_image: Optional[List[Union[str, Dict]]] = None
            if image:
                base64_image = await process_image(image)

            if files:
                for file in files:
                    contents = await file.read()
                    file_obj = BytesIO(contents)
                    file_obj.name = file.filename

                    if file.content_type == "application/pdf":
                        from phi.document.reader.pdf import PDFReader
                        file_content = PDFReader().read(file_obj)
                    elif file.content_type == "text/csv":
                        from phi.document.reader.csv_reader import CSVReader
                        file_content = CSVReader().read(file_obj)
                    elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        from phi.document.reader.docx import DocxReader
                        file_content = DocxReader().read(file_obj)
                    elif file.content_type == "text/plain":
                        from phi.document.reader.text import TextReader
                        file_content = TextReader().read(file_obj)
                    else:
                        raise HTTPException(status_code=400, detail="Unsupported file type")

                    if agent.knowledge:
                        agent.knowledge.load_documents(file_content)

            if stream:
                return StreamingResponse(
                    chat_response_streamer(new_agent_instance, message, images=base64_image),
                    media_type="text/event-stream",
                )
            else:
                run_response = cast(
                    RunResponse,
                    await new_agent_instance.arun(message, images=base64_image, stream=False),
                )
                return run_response.model_dump_json()

        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(traceback.format_exc())
            return JSONResponse(status_code=500, content={"message": str(e)})

    @playground_router.post("/agent/sessions/all")
    async def get_agent_sessions(request: Request, body: AgentSessionsRequest):
        try:
            user_id = request.state.user.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="Authentication required")

            logger.debug(f"AgentSessionsRequest: {body} | User: {user_id}")

            agent = get_agent_by_id(body.agent_id, agents)
            if agent is None:
                return JSONResponse(status_code=404, content={"message": "Agent not found."})

            if agent.storage is None:
                return JSONResponse(status_code=404, content={"message": "Agent does not have storage enabled."})

            agent_sessions: List[AgentSessionsResponse] = []
            all_agent_sessions: List[AgentSession] = agent.storage.get_all_sessions(user_id=user_id)

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

        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(traceback.format_exc())
            return JSONResponse(status_code=500, content={"message": str(e)})

    @playground_router.post("/agent/sessions/{session_id}")
    async def get_agent_session(request: Request, session_id: str, body: AgentSessionsRequest):
        try:
            user_id = request.state.user.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="Authentication required")

            logger.debug(f"AgentSessionsRequest: {body} | User: {user_id}")

            agent = get_agent_by_id(body.agent_id, agents)
            if agent is None:
                return JSONResponse(status_code=404, content={"message": "Agent not found."})

            if agent.storage is None:
                return JSONResponse(status_code=404, content={"message": "Agent does not have storage enabled."})

            agent_session: Optional[AgentSession] = agent.storage.read(session_id, user_id)
            if agent_session is None:
                return JSONResponse(status_code=404, content={"message": "Session not found."})

            return agent_session

        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(traceback.format_exc())
            return JSONResponse(status_code=500, content={"message": str(e)})


    @playground_router.post("/agent/session/rename")
    async def agent_rename(request: Request, body: AgentRenameRequest):
        try:
            user_id = request.state.user.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="Authentication required")

            agent = get_agent_by_id(body.agent_id, agents)
            if agent is None:
                return JSONResponse(status_code=404, content={"message": f"Couldn't find agent with ID {body.agent_id}"})

            logger.debug(f"User {user_id} renaming session {body.session_id} to {body.name}")

            agent.session_id = body.session_id
            agent.rename_session(body.name)
            return JSONResponse(content={"message": f"Successfully renamed agent session to {body.name}"})

        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(traceback.format_exc())
            return JSONResponse(status_code=500, content={"message": str(e)})

    @playground_router.post("/agent/session/delete")
    async def agent_session_delete(request: Request, body: AgentSessionDeleteRequest):
        try:
            user_id = request.state.user.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="Authentication required")

            agent = get_agent_by_id(body.agent_id, agents)
            if agent is None:
                return JSONResponse(status_code=404, content={"message": "Agent not found."})

            if agent.storage is None:
                return JSONResponse(status_code=404, content={"message": "Agent does not have storage enabled."})

            logger.debug(f"User {user_id} requested deletion of session {body.session_id}")

            all_agent_sessions: List[AgentSession] = agent.storage.get_all_sessions(user_id=user_id)
            for session in all_agent_sessions:
                if session.session_id == body.session_id:
                    agent.delete_session(body.session_id)
                    return JSONResponse(content={"message": f"Successfully deleted session {body.session_id} for agent {agent.name}"})

            return JSONResponse(status_code=404, content={"message": "Session not found."})

        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(traceback.format_exc())
            return JSONResponse(status_code=500, content={"message": str(e)})

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
