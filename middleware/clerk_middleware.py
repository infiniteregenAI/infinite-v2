import os
import traceback
import logging

from fastapi_clerk_auth import ClerkConfig, ClerkHTTPBearer
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

clerk_config = ClerkConfig(jwks_url=os.getenv("CLERK_JWKS_URL"))
clerk_auth_guard = ClerkHTTPBearer(config=clerk_config)

async def clerk_middleware(request: Request, call_next):
    """
    Middleware to authenticate requests using Clerk
    
    Args:
        request (Request): Incoming request
    
    Returns:
        Response: JSONResponse object
    """
    try:
        # Allow access to public routes like /docs
        if request.url.path == "/docs" or request.url.path == "/openapi.json":
            response = await call_next(request)
            return response

        auth_header = request.headers.get("Authorization")
        if auth_header:
            try:
                token = await clerk_auth_guard(request)
                request.state.user = token.decoded
            except HTTPException:
                return JSONResponse(status_code=403, content={"detail": "Invalid or expired token"})
        else:
            return JSONResponse(status_code=403, content={"detail": "Authorization header missing"})

        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Error in clerk_middleware: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail": f"Internal server error {e}"})
