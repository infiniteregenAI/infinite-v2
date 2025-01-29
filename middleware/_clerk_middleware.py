import os 
import logging

from fastapi import  Request
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from clerk_backend_api.jwks_helpers import AuthenticateRequestOptions
from clerk_backend_api import Clerk

from utils.constants import CLERK_ALLOWED_PARTIES

load_dotenv()
logger = logging.getLogger(__name__)

CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
clerk_client = Clerk(bearer_auth=CLERK_SECRET_KEY)

class ClerkAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            excluded_paths = [""]
            
            if request.url.path in excluded_paths:
                logger.info(f"Skipping ClerkAuthMiddleware for {request.url.path}")
                return await call_next(request)
            
            request_body = await request.body()
            logger.info(f"Incoming request headers: {request.headers}, request body: {request_body}")
            try:
                request_state = clerk_client.authenticate_request(
                    request,
                    AuthenticateRequestOptions(
                        authorized_parties=CLERK_ALLOWED_PARTIES
                    )
                )
                if not request_state.is_signed_in:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Unauthorized access"}
                    )
                request.state.user = request_state.user
            except Exception:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or expired token"}
                )

            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Error in ClerkAuthMiddleware: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )