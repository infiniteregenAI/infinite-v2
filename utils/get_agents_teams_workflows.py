from utils.reserved_agents import climate_ai, green_pill_ai, owocki_ai, gitcoin_ai, reserved_agents_team
from workflows.blog_post_generator import *
from workflows.research_workflow import *

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

def load_all_agents_n_teams():
    """
    Load all agents 
    
    Returns:
        list: List of agents
    """
    with open("agents.json", "r") as agent_file:
        agents_data = json.load(agent_file)

    dynamic_agents = []
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
        dynamic_agents.append(dynamic_agent)
        
    all_agents = [
        climate_ai,
        green_pill_ai,
        owocki_ai,
        gitcoin_ai,
    ] + dynamic_agents
    
    with open("teams.json", "r") as teams_file:
        teams_data = json.load(teams_file)

    dynamic_teams = []
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
        dynamic_teams.append(team_agent)
    
    test_team = Agent(
        agent_id="test-team",
        name="Test Team",
        team=dynamic_agents[-3:],
        instructions="Test Team instructions",
        role="Team",
        description="Test Team description",
        task="Test Team task",
        stream=False,
        show_tool_calls=True,
    )
        
    all_teams = [
        reserved_agents_team,test_team
    ] + dynamic_teams 
    
    return all_agents , all_teams  