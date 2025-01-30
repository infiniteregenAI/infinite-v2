from typing import List, Dict, Any
import json
import os
import traceback
import logging 
import sqlite3
import pickle

from phi.agent import Agent
from phi.tools.hackernews import HackerNews
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from dotenv import load_dotenv

from schemas.agents_schema import AgentResponse
from schemas.teams_schema import TeamResponse
from phi_server import playground_instance

logger = logging.getLogger(__name__)
load_dotenv()

DB_URL = os.getenv("DB_URL")

class AgentManager:
    """
    Manages the creation, storage, and retrieval of agents and teams.
    
    Attributes:
        storage_path (str): Path to the JSON file storing agent data
        team_storage_path (str): Path to the JSON file storing team data
    """
    
    def __init__(self, storage_path: str = "agents.json", team_storage_path: str = "teams.json"):
        """
        Initialize the AgentManager.
        
        Args:
            storage_path (str): Path for storing agent data
            team_storage_path (str): Path for storing team data
        """
        self.storage_path = storage_path
        self.team_storage_path = team_storage_path
        self._ensure_storage_exists()
        self.agent_db_con = sqlite3.connect('tmp/agents.db')


    def _ensure_storage_exists(self) -> None:
        """Create storage files if they don't exist."""
        for path in [self.storage_path, self.team_storage_path]:
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    json.dump([], f)

    def _initialize_tool(self, tool_name: str) -> Any:
        """
        Initialize a tool instance based on the tool name.
        
        Args:
            tool_name (str): Name of the tool to initialize
            
        Returns:
            Tool instance or None if tool not found
        """
        tool_map = {
            "HackerNews": HackerNews(),
            "DuckDuckGo": DuckDuckGo(),
            "Newspaper4k": Newspaper4k(),
        }
        return tool_map.get(tool_name)
    
    def get_agents_by_user(self, user_id: str) -> List[AgentResponse]:
        """
        Retrieve all agents created by a specific user.
        
        Args:
            user_id (str): The ID of the user whose agents to retrieve
            
        Returns:
            List[AgentResponse]: List of agents belonging to the user
        """
        agents = self.get_all_agents()
        user_agents = [agent for agent in agents if agent.get('user_id') == user_id]
        return [AgentResponse(**agent) for agent in user_agents]

    def create_agent(self, name: str, role: str, tools: List[str], 
                description: str, instructions: List[str], 
                urls: List[str] = None, user_id: str = None) -> AgentResponse:
        """
        Create a new agent and save it to storage.
        
        Args:
            name (str): Name of the agent
            role (str): Role of the agent
            tools (List[str]): List of tool names the agent can use
            description (str): Description of the agent
            instructions (List[str]): List of instructions for the agent
            urls (List[str], optional): List of URLs for the agent to use
            user_id (str): ID of the user creating the agent
            
        Returns:
            AgentResponse: Created agent data
        """
        # Get existing agents and generate new ID
        agents = self.get_all_agents()
        agent_id = str(len(agents) + 1)

        # Initialize urls if None
        urls = urls or []
        
        # Create response model
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
        agent = Agent(
            name=name,
            role=role,
            tools=[self._initialize_tool(tool_name) for tool_name in tools],
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
        
        playground_instance.add_agent(agent)

        
        agents.append(agent_data.dict())
        with open(self.storage_path, 'w') as f:
            json.dump(agents, f, indent=2)
            
        return agent_data

    def get_all_agents(self) -> List[Dict]:
        """
        Retrieve all agents from storage.
        
        Returns:
            List[Dict]: List of all stored agents
        """
        with open(self.storage_path, 'r') as f:
            return json.load(f)

    def get_agent_by_id(
        self, 
        agent_id: str,
    ) -> Agent:
        """
        Get an agent by its ID.
        
        Args:
            agent_id (str): ID of the agent to retrieve
            
        Returns:
            Agent: Agent instance or error message if agent not found or an error occurs during initialization      
        """
        try:
            agents = self.get_all_agents()
            agent_data = next((a for a in agents if a['id'] == agent_id), None)
            
            if not agent_data:
                raise ValueError(f"Agent with ID '{agent_id}' not found")
                        
            cursor = self.agent_db_con.cursor()
            cursor.execute('SELECT agent FROM agents WHERE id = ?', (f'agent_{agent_id}',))
            result = cursor.fetchone()

            if result:
                serialized_agent = result[0] 
                agent = pickle.loads(serialized_agent) 
            else:
                raise ValueError(f"Agent with ID '{agent_id}' not found")
                            
            return agent
        except Exception as e:
            logger.error(traceback.format_exc())
            return {"message": str(e)}
    
    def create_team(
        self,  
        user_id:str,
        name: str, 
        agent_names: List[str], 
        instructions: List[str]
    ) -> TeamResponse:
        """
        Create a new team from existing agents.
        
        Args:
            user_id (str): User ID of the team creator
            name (str): Name of the team
            agent_names (List[str]): List of agent names to include in the team
            instructions (List[str]): List of instructions for the team
            
        Returns:
            TeamResponse: Created team data
            
        Raises:
            ValueError: If any agent is not found or if less than two agents are provided
        """
        if len(agent_names) < 2:
            raise ValueError("At least two agents are required to create a team.")
        
        teams = self.get_all_teams(
            user_id=user_id
        )
        team_id = str(len(teams) + 1)

        agent_responses = []
        
        with open(self.storage_path, 'r') as f:
            agents = json.load(f)
            
        for agent_name in agent_names:
            agent_data = next((a for a in agents if a['name'] == agent_name and a['user_id'] == user_id ), None)
            if not agent_data:
                raise ValueError(f"Agent with name '{agent_name}' not found")
            agent_responses.append(AgentResponse(**agent_data))
        
        team_data = TeamResponse(
            id=team_id,
            user_id=user_id,
            name=name,
            agents=agent_responses,
            instructions=instructions,
            markdown=True,
            show_tool_calls=True
        )
        teams.append(team_data.dict())
        with open(self.team_storage_path, 'w') as f:
            json.dump(teams, f, indent=2)   
        return team_data

    def get_all_teams(
        self,
        user_id: str
    ) -> List[Dict]:
        """
        Retrieve all teams from storage.
        
        Args:
            user_id (str): User ID to filter teams by
        
        Returns:
            List[Dict]: List of all stored teams
        """
        teams = None
        with open(self.team_storage_path, 'r') as f:
            teams = json.load(f)
        
        return [t for t in teams if t['user_id'] == user_id]

    def get_team_by_id(
        self, 
        team_id: str,
        user_id: str,
        session_id: str
    ):
        try:
            
            teams = self.get_all_teams(user_id)
            team_data = next((t for t in teams if t['id'] == team_id), None)
            
            if not team_data:
                raise ValueError(f"Team with ID '{team_id}' not found")
            
            agents = [self.get_agent_by_id(agent_id) for agent_id in team_data['agents']]
            session_storage = SqlAgentStorage(table_name=f"{team_id}_ai_sessions", db_file="tmp/teams_sessions.db")
            
            team = Agent(
                agent_id=team_data['id'],
                name=team_data['name'],
                team=agents,
                instructions=team_data['instructions'],
                show_tool_calls=team_data['show_tool_calls'],
                markdown=team_data['markdown'],
                session_id=session_id,
                user_id=user_id,
                storage=session_storage,
                add_history_to_messages=True,
                num_history_responses=5,
            )
            return team
        except Exception as e:
            logger.error(traceback.format_exc())
            return {"message": str(e)}
    
    def update_agent(self, name: str, role: str, tools: List[str], 
                    description: str, instructions: List[str],
                    urls: List[str] = None) -> AgentResponse:
        """
        Update an existing agent's details.
        
        Args:
            name (str): Name of the agent to update
            role (str): New role of the agent
            tools (List[str]): Updated list of tool names the agent can use
            description (str): Updated description of the agent
            instructions (List[str]): Updated list of instructions for the agent
            urls (List[str], optional): Updated list of URLs for the agent to use
            
        Returns:
            AgentResponse: Updated agent data
        """
        agents = self.get_all_agents()
        agent_data = next((a for a in agents if a['name'] == name), None)
        
        if not agent_data:
            raise ValueError(f"Agent with name '{name}' not found")
        
        # Update agent details
        agent_data['role'] = role
        agent_data['tools'] = tools
        agent_data['description'] = description
        agent_data['instructions'] = instructions
        agent_data['urls'] = urls or []

        # Save updated agents list to storage
        with open(self.storage_path, 'w') as f:
            json.dump(agents, f, indent=2)

        return AgentResponse(**agent_data)

