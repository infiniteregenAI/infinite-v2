import os 

from dotenv import load_dotenv
from phi.playground import Playground, serve_playground_app
from fastapi.middleware.cors import CORSMiddleware

from utils.get_agents_teams_workflows import load_all_agents , load_all_workflows , load_all_teams
from middlewares.clerk_middleware import ClerkAuthMiddleware
from schemas.database import init_db
from utils.constants import ALLOWED_ORIGINS

load_dotenv()
init_db()

all_agents = load_all_agents()
all_teams = load_all_teams()
all_workflows = load_all_workflows()

playground_instance = Playground(
    agents=all_agents+all_teams, 
    workflows=all_workflows,
)

playground = playground_instance.get_app()
playground.add_middleware(ClerkAuthMiddleware, api_key=os.getenv("CLERK_SECRET_KEY"))
playground.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, 
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"], 
)


if __name__ == "__main__":
    serve_playground_app("phi_server:playground", host="0.0.0.0", reload=True)