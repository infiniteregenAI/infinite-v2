from utils.reserved_agents import climate_ai, green_pill_ai, owocki_ai, gitcoin_ai, reserved_agents_team
from workflows.blog_post_generator import *
from workflows.research_workflow import *
from schemas.database import get_db_session, DatabaseOperations, TeamOperations, TeamDB, AgentDB
import json

generate_blog_post = BlogPostGenerator(
        session_id=f"generate-blog-post-on",
        storage=SqlWorkflowStorage(
            table_name="generate_blog_post_workflows",
            db_file="tmp/workflows.db",
        ),
    )

research_workflow = ResearchWorkflow(
    session_id="research-workflow",
    storage=SqlWorkflowStorage(
        table_name="research_workflows",
        db_file="tmp/workflows.db",
    ),
)

def load_all_workflows():
    """
    Load all workflows
    
    Returns:
        list: List of workflows
    """
    return [generate_blog_post, research_workflow]

def load_all_agents():
    """
    Load all agents from both database and file system
    
    Returns:
        list: List of agents
    """
    all_agents = [
        climate_ai,
        green_pill_ai,
        owocki_ai,
        gitcoin_ai,
    ]
    
    try:
        with open("agents.json", "r") as agent_file:
            agents_data = json.load(agent_file)

        for agent_data in agents_data:
            dynamic_agent = Agent(
                name=agent_data["name"],
                agent_id=agent_data["id"],
                user_id=agent_data["user_id"],
                introduction=f"I am {agent_data['name']}, a {agent_data['role']}.",
                role=agent_data["role"],
                description=agent_data["description"],
                instructions=agent_data["instructions"],
                task=f"Perform the tasks outlined in the instructions for {agent_data['name']}.",
                search_knowledge=True,
                stream=agent_data.get("markdown", False),
            )
            all_agents.append(dynamic_agent)
            
            try:
                with get_db_session() as db:
                    existing_agent = db.query(AgentDB).filter(AgentDB.id == agent_data["id"]).first()
                    if not existing_agent:
                        DatabaseOperations.create_agent(db, agent_data)
            except Exception as db_error:
                print(f"Failed to sync agent to database: {db_error}")
                
    except Exception as file_error:
        print(f"File-based loading failed: {file_error}")

    try:
        with get_db_session() as db:
            db_agents = DatabaseOperations.get_agents_by_user(db, user_id="*")
            if db_agents:
                for agent_data in db_agents:
                    if not any(a.agent_id == agent_data.id for a in all_agents):
                        dynamic_agent = Agent(
                            name=agent_data.name,
                            agent_id=agent_data.id,
                            user_id=agent_data.user_id,
                            introduction=f"I am {agent_data.name}, a {agent_data.role}.",
                            role=agent_data.role,
                            description=agent_data.description,
                            instructions=agent_data.instructions,
                            task=f"Perform the tasks outlined in the instructions for {agent_data.name}.",
                            search_knowledge=True,
                            stream=agent_data.markdown,
                        )
                        all_agents.append(dynamic_agent)
    except Exception as e:
        print(f"Database loading failed: {e}")
    
    return all_agents

def load_all_teams():
    """
    Load all teams from both database and file system
    
    Returns:
        list: List of teams
    """
    all_teams = [reserved_agents_team]
    
    try:
        with open("teams.json", "r") as teams_file:
            teams_data = json.load(teams_file)

        for team_data in teams_data:
            team_agents = []
            for agent_data in team_data["agents"]:
                team_agent = Agent(
                    agent_id=agent_data["id"],
                    name=agent_data["name"],
                    introduction=f"I am {agent_data['name']}, a {agent_data['role']}.",
                    role=agent_data["role"],
                    description=agent_data["description"],
                    instructions=agent_data["instructions"],
                    task=f"Perform the tasks outlined in the instructions for {agent_data['name']}.",
                    search_knowledge=True,
                    stream=agent_data.get("markdown", False),
                )
                team_agents.append(team_agent)
    
            team_agent = Agent(
                agent_id=team_data["id"],
                name=team_data["name"],
                team=team_agents,
                instructions=team_data["instructions"],
                role="Team",
                description=f"The {team_data['name']} consists of multiple agents collaborating to achieve shared goals.",
                task=f"Collaborate as a team to achieve the goals specified for {team_data['name']}.",
                stream=team_data.get("markdown", False),
                show_tool_calls=team_data.get("show_tool_calls", True),
            )
            all_teams.append(team_agent)
            
            try:
                with get_db_session() as db:
                    existing_team = db.query(TeamDB).filter(TeamDB.id == team_data["id"]).first()
                    if not existing_team:
                        TeamOperations.create_team(db, {
                            "id": team_data["id"],
                            "name": team_data["name"],
                            "description": team_data.get("description"),
                            "role": "Team",
                            "instructions": team_data["instructions"],
                            "owner_id": team_data.get("owner_id", "system"),
                            "agent_ids": [a["id"] for a in team_data["agents"]],
                            "is_active": True
                        })
            except Exception as db_error:
                print(f"Failed to sync team to database: {db_error}")

    except Exception as file_error:
        print(f"File-based loading failed: {file_error}")

    try:
        with get_db_session() as db:
            db_teams = TeamOperations.get_teams_by_user(db, user_id="*")
            if db_teams:
                for team_data in db_teams:
                    if not any(t.agent_id == team_data.id for t in all_teams):
                        team_agents = []
                        for agent_id in team_data.agent_ids:
                            agent = DatabaseOperations.get_agent(db, agent_id)
                            if agent:
                                team_agent = Agent(
                                    agent_id=agent.id,
                                    name=agent.name,
                                    introduction=f"I am {agent.name}, a {agent.role}.",
                                    role=agent.role,
                                    description=agent.description,
                                    instructions=agent.instructions,
                                    task=f"Perform the tasks outlined in the instructions for {agent.name}.",
                                    search_knowledge=True,
                                    stream=agent.markdown,
                                )
                                team_agents.append(team_agent)
                        
                        team_agent = Agent(
                            agent_id=team_data.id,
                            name=team_data.name,
                            team=team_agents,
                            instructions=team_data.instructions,
                            role="Team",
                            description=team_data.description,
                            task=f"Collaborate as a team to achieve the goals specified for {team_data.name}.",
                            stream=True,
                            show_tool_calls=True,
                        )
                        all_teams.append(team_agent)
    except Exception as e:
        print(f"Database loading failed: {e}")
    
    return all_teams