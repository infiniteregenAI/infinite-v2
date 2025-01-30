import httpx
import os 

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from jose import jwt, jwk
from dotenv import load_dotenv

load_dotenv()

JWKS_JSON = os.getenv("JWKS_JSON")
JWKS_ISSUER = os.getenv("JWKS_ISSUER")

class ClerkAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str):
        """
        Middleware to validate Clerk JWT tokens
        
        Args:
            app: FastAPI app instance
            api_key: Clerk API key
            
        Returns:
            None
        """
        super().__init__(app)
        self.api_key = api_key
        self.clerk_jwt_public_key = None

    async def fetch_jwt_public_key(self):
        """
        Fetch the public key from the JWKS endpoint
        
        Returns:
            jwk.JWK: Public key object
        """
        if not self.clerk_jwt_public_key:
            async with httpx.AsyncClient() as client:
                response = await client.get(JWKS_JSON)
                jwks = response.json()
                key_data = jwks['keys'][0]  # Assuming the first key is valid
                self.clerk_jwt_public_key = jwk.construct(key_data)
        return self.clerk_jwt_public_key

    async def verify_token(self, token: str) -> dict:
        """
        Verify the JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            dict: Claims from the token payload if valid or raises an HTTPException
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
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

    async def dispatch(self, request: Request, call_next):
        """
        Middleware dispatch method
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in the chain
            
        Returns:
            Response from the next middleware in the chain or an HTTPException
        """
        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing authentication token")

        token = authorization.split("Bearer ")[1]
        claims = await self.verify_token(token)
        request.state.user = claims 

        return await call_next(request)