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

    async def verify_token(self, token: str) -> dict:
        """
        Verify the JWT token
        
        Args:
            token (str): JWT token
            
        Returns:
            dict: Claims in the JWT token
        """
        try:
            header = jwt.get_unverified_header(token)
            public_key = await self.fetch_jwt_public_key()
            payload = jwt.decode(
                token,
                public_key.to_pem().decode('utf-8'),
                algorithms=["RS256"],
                options={"verify_aud": False},
                issuer=JWKS_ISSUER
            )
            return payload
        except Exception as e:
            logger.error(f"Error verifying JWT token: {traceback.format_exc()}")
            return JSONResponse(status_code=401, content={"detail": "Invalid JWT token"})

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
            if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
                return await call_next(request)

            authorization = request.headers.get("Authorization")
            if not authorization or not authorization.startswith("Bearer "):
                return JSONResponse(status_code=401, content={"detail": "Missing Authorization header with Bearer token"})

            token = authorization.split("Bearer ")[1]
            claims = await self.verify_token(token)
            request.state.user = claims  

            return await call_next(request)
        except Exception as e:
            logger.error(f"Error in ClerkAuthMiddleware: {traceback.format_exc()}")
            return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})