import logging
import os 

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import agents_router, teams_router, tools_router
from middlewares.clerk_middleware import ClerkAuthMiddleware
from utils.constants import ALLOWED_ORIGINS

load_dotenv()

CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  
    ]
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, 
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"], 
)

# app.add_middleware(ClerkAuthMiddleware, api_key=CLERK_SECRET_KEY)

@app.get("/")
async def root():
    return {"message": "Welcome to the Backend"}

app.include_router(agents_router.router, prefix="/api/v1", tags=["agents"])
app.include_router(teams_router.router, prefix="/api/v1", tags=["teams"])
app.include_router(tools_router.router, prefix="/api/v1", tags=["tools"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)