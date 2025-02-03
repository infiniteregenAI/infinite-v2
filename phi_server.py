import os 

from dotenv import load_dotenv
from phi.playground import Playground, serve_playground_app
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from middlewares.clerk_middleware import ClerkAuthMiddleware

from utils.get_agents_teams_workflows import load_all_agents , load_all_workflows , load_all_teams
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

# Add auth middleware after CORS

playground.add_middleware(ClerkAuthMiddleware, api_key=os.getenv("CLERK_SECRET_KEY"))
# @playground.middleware("https")
# async def clerk_auth_middleware(request: Request, call_next):
#     """
#     Middleware to authenticate users via Clerk JWT.
#     """
#     if request.url.path in ["/docs", "/redoc", "/openapi.json"]:  # Skip auth for docs
#         return await call_next(request)

#     authorization = request.headers.get("Authorization")
#     if not authorization or not authorization.startswith("Bearer "):
#         return JSONResponse(status_code=401, content={"detail": "Missing or invalid Authorization header"})

#     token = authorization.split("Bearer ")[1]
#     claims = await verify_token(token)

#     if not claims:
#         return JSONResponse(status_code=401, content={"detail": "Invalid token"})

#     request.state.user = claims  # Attach user claims to request
#     return await call_next(request)

if __name__ == "__main__":
    serve_playground_app("phi_server:playground", host="0.0.0.0", reload=True)