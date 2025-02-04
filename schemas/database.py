from sqlalchemy import create_engine, Column, String, Boolean, ARRAY, JSON, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator, List, Optional
from pydantic import BaseModel
import os
from fastapi import HTTPException

# Database configuration
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise ValueError("Database URL not found in environment variables")

# Create engine with proper configuration
engine = create_engine(
    DB_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AgentCreate(BaseModel):
    name: str
    role: str
    tools: Optional[List[str]] = []
    description: Optional[str] = None
    instructions: Optional[str] = None
    pdf_urls: Optional[List[str]] = []
    website_urls: Optional[List[str]] = []
    markdown: bool = True
    show_tool_calls: bool = True
    add_datetime_to_instructions: bool = True
    user_id: str

class AgentDB(Base):
    __tablename__ = "agents"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)
    tools = Column(ARRAY(String), nullable=True, server_default='{}')
    description = Column(String, nullable=True)
    instructions = Column(ARRAY(String), nullable=True, server_default='{}')
    pdf_urls = Column(ARRAY(String), nullable=True, server_default='{}')
    website_urls = Column(ARRAY(String), nullable=True, server_default='{}')
    markdown = Column(Boolean, default=True)
    show_tool_calls = Column(Boolean, default=True)
    add_datetime_to_instructions = Column(Boolean, default=True)
    user_id = Column(String, nullable=False, index=True)

class TeamDB(Base):
    __tablename__ = "teams"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(String, nullable=True)
    role = Column(String, nullable=True)
    instructions = Column(ARRAY(String), nullable=True, server_default='{}')
    tools = Column(ARRAY(String), nullable=True, server_default='{}')
    owner_id = Column(String, nullable=False, index=True)
    agent_ids = Column(ARRAY(String), nullable=True, server_default='{}')
    is_active = Column(Boolean, default=True)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def init_db():
    inspector = inspect(engine)
    
    # Check if tables already exist
    existing_tables = inspector.get_table_names()
    
    if "agents" not in existing_tables or "teams" not in existing_tables:
        try:
            Base.metadata.create_all(bind=engine)
            print("Database tables created successfully")
        except Exception as e:
            print(f"Error creating database tables: {e}")
            raise
    else:
        print("Database tables already exist, skipping initialization")

class DatabaseOperations:
    @staticmethod
    def create_agent(db: Session, agent_data: dict) -> AgentDB:
        try:
            db_agent = AgentDB(**agent_data)
            db.add(db_agent)
            db.commit()
            db.refresh(db_agent)
            return db_agent
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")

    @staticmethod
    def get_agent(db: Session, agent_id: str) -> Optional[AgentDB]:
        agent = db.query(AgentDB).filter(AgentDB.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent with id {agent_id} not found")
        return agent

    @staticmethod
    def get_agents_by_user(db: Session, user_id: str) -> List[AgentDB]:
        try:
            return db.query(AgentDB).filter(AgentDB.user_id == user_id).all()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch agents: {str(e)}")

    @staticmethod
    def update_agent(db: Session, agent_id: str, agent_data: dict) -> AgentDB:
        try:
            db_agent = DatabaseOperations.get_agent(db, agent_id)
            
            for key, value in agent_data.items():
                if hasattr(db_agent, key):
                    setattr(db_agent, key, value)
            
            db.commit()
            db.refresh(db_agent)
            return db_agent
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to update agent: {str(e)}")

    @staticmethod
    def delete_agent(db: Session, agent_id: str) -> bool:
        try:
            db_agent = DatabaseOperations.get_agent(db, agent_id)
            db.delete(db_agent)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")
    
class TeamOperations:
    @staticmethod
    def create_team(db: Session, team_data: dict) -> TeamDB:
        try:
            db_team = TeamDB(**team_data)
            db.add(db_team)
            db.commit()
            db.refresh(db_team)
            return db_team
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to create team: {str(e)}")

    @staticmethod
    def get_team(db: Session, team_id: str) -> Optional[TeamDB]:
        team = db.query(TeamDB).filter(TeamDB.id == team_id).first()
        if not team:
            raise HTTPException(status_code=404, detail=f"Team with id {team_id} not found")
        return team

    @staticmethod
    def get_teams_by_user(db: Session, user_id: str) -> List[TeamDB]:
        try:
            return db.query(TeamDB).filter(TeamDB.owner_id == user_id).all()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch teams: {str(e)}")

    @staticmethod
    def update_team(db: Session, team_id: str, team_data: dict) -> TeamDB:
        try:
            db_team = TeamOperations.get_team(db, team_id)
            
            for key, value in team_data.items():
                if hasattr(db_team, key):
                    setattr(db_team, key, value)
            
            db.commit()
            db.refresh(db_team)
            return db_team
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to update team: {str(e)}")

    @staticmethod
    def delete_team(db: Session, team_id: str) -> bool:
        try:
            db_team = TeamOperations.get_team(db, team_id)
            db.delete(db_team)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to delete team: {str(e)}")