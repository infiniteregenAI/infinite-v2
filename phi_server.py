import os 

from dotenv import load_dotenv
from phi.playground import Playground, serve_playground_app
from fastapi.middleware.cors import CORSMiddleware
from middlewares.clerk_middleware import ClerkAuthMiddleware

from utils.get_agents_teams_workflows import load_all_agents_n_teams , load_all_workflows 
from schemas.database import init_db
from utils.constants import ALLOWED_ORIGINS

load_dotenv()
init_db()

all_agents , all_teams = load_all_agents_n_teams()
all_workflows = load_all_workflows()

playground_instance = Playground(
    agents=all_agents+all_teams, 
    workflows=all_workflows,
)

playground = playground_instance.get_app()

# Add CORS middleware before the auth middleware
playground.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

playground.add_middleware(ClerkAuthMiddleware, api_key=os.getenv("CLERK_SECRET_KEY"))

if __name__ == "__main__":
    serve_playground_app("phi_server:playground", host="0.0.0.0", reload=True)