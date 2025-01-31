import httpx
import os 
import traceback
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from jose import jwt, jwk
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

JWKS_JSON = os.getenv("JWKS_JSON")
JWKS_ISSUER = os.getenv("JWKS_ISSUER")

class ClerkAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str):
        """
        Middleware for verifying JWT tokens from Clerk
        
        Args:
            app (FastAPI): FastAPI instance
            api_key (str): Clerk API Key
            
        Returns:
            ClerkAuthMiddleware: Instance of ClerkAuthMiddleware
        """
        try:
            super().__init__(app)
            self.api_key = api_key
            self.clerk_jwt_public_key = None
            
        except Exception as e:
            logger.error(f"Error initializing ClerkAuthMiddleware: {traceback.format_exc()}")
            return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

    async def fetch_jwt_public_key(self):
        """
        Fetch the JWT public key from Clerk
        
        Returns:
            jwk.JWK: Public key for verifying JWT tokens
        """
        try:
            if not self.clerk_jwt_public_key:
                async with httpx.AsyncClient() as client:
                    response = await client.get(JWKS_JSON)
                    jwks = response.json()
                    key_data = jwks['keys'][0]  
                    self.clerk_jwt_public_key = jwk.construct(key_data)
            return self.clerk_jwt_public_key
        
        except Exception as e:
            logger.error(f"Error fetching JWT public key: {traceback.format_exc()}")
            return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

    async def verify_token(self, token: str) -> dict | None:
        """
        Verify the JWT token

        Args:
            token (str): JWT token

        Returns:
            dict: Claims in the JWT token, or None if verification fails
        """
        try:
            header = jwt.get_unverified_header(token)
            public_key = await self.fetch_jwt_public_key()
            payload = jwt.decode(
                token,
                public_key.to_pem().decode("utf-8"),
                algorithms=["RS256"],
                options={"verify_aud": False},
                issuer=JWKS_ISSUER,
            )
            return payload 
        except Exception as e:
            logger.error(f"Error verifying JWT token: {traceback.format_exc()}")
            return None  
        
    async def dispatch(self, request: Request, call_next):
        """
        Dispatch method for the middleware

        Args:
            request (Request): Request object
            call_next (Callable): Next callable

        Returns:
            Response: Response object from the middleware
        """
        try:
            # Log the request path and headers for debugging
            logger.debug(f"Request path: {request.url.path}")
            logger.debug(f"Request headers: {dict(request.headers)}")

            # Allow OPTIONS requests to pass through without authentication
            if request.method == "OPTIONS":
                response = await call_next(request)
                return response

            # Skip authentication for docs
            if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
                return await call_next(request)

            # Check both Authorization and authorization headers
            authorization = request.headers.get("Authorization") or request.headers.get("authorization")
            
            if not authorization:
                logger.debug("No Authorization header found")
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing Authorization header with Bearer token"},
                )

            if not authorization.startswith("Bearer "):
                logger.debug("Authorization header doesn't start with Bearer")
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid Authorization header format"},
                )

            token = authorization.split("Bearer ")[1].strip()
            if not token:
                logger.debug("Empty token found")
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Empty token provided"},
                )

            claims = await self.verify_token(token)
            logger.debug(f"Token claims: {claims}")

            if not claims:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
            
            request.state.user = claims  
            return await call_next(request)
        except Exception as e:
            logger.error(f"Error in ClerkAuthMiddleware: {traceback.format_exc()}")
            return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
