#!/usr/bin/env python3
"""
MISO Security Configuration - CORS & WebSocket Hardening
=========================================================

Production-ready security configuration for MISO FastAPI backend.
Implements allowlist-based CORS, WebSocket origin validation, and rate limiting.

Environment Variables:
    ALLOWED_ORIGINS: Comma-separated list of allowed origins (default: localhost only)
    API_KEY: API key for authentication (required in production)  
    RATE_LIMIT_REQUESTS: Requests per minute (default: 100)
    RATE_LIMIT_WINDOW: Rate limit window in seconds (default: 60)
"""

import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Security configuration
class SecurityConfig:
    def __init__(self):
        self.allowed_origins = self._parse_allowed_origins()
        self.api_key = os.getenv("API_KEY")
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        # Warn if no API key in production
        if not self.api_key and os.getenv("ENVIRONMENT") == "production":
            logger.warning("⚠️ No API_KEY set in production environment!")

    def _parse_allowed_origins(self) -> List[str]:
        """Parse allowed origins from environment."""
        default_origins = "http://127.0.0.1:5151,http://localhost:5151,http://127.0.0.1:8000,http://localhost:8000"
        origins_str = os.getenv("ALLOWED_ORIGINS", default_origins)
        origins = [origin.strip() for origin in origins_str.split(",")]
        
        logger.info(f"Allowed origins: {origins}")
        return origins

# Global security config
security_config = SecurityConfig()

# Rate limiter with Redis backend for distributed setups
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=security_config.redis_url,
    default_limits=[f"{security_config.rate_limit_requests}/minute"]
)

# Bearer token authentication
security = HTTPBearer(auto_error=False)

def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    """Verify API key from Bearer token."""
    if not security_config.api_key:
        return True  # No auth required if no key set
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != security_config.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return True

def add_cors_middleware(app: FastAPI) -> None:
    """Add CORS middleware with security hardening."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=security_config.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=[
            "Authorization", 
            "Content-Type",
            "X-Requested-With",
            "X-API-Key"
        ],
        expose_headers=["X-Rate-Limit-Remaining", "X-Rate-Limit-Reset"],
    )
    logger.info("✅ CORS middleware configured with allowlist")

def add_rate_limiting(app: FastAPI) -> None:
    """Add rate limiting middleware."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info(f"✅ Rate limiting configured: {security_config.rate_limit_requests}/min")

async def verify_websocket_origin(websocket: WebSocket) -> None:
    """Verify WebSocket connection origin."""
    origin = websocket.headers.get("origin")
    
    if not origin:
        logger.warning(f"WebSocket connection without origin header from {websocket.client}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
    
    if origin not in security_config.allowed_origins:
        logger.warning(f"WebSocket connection from unauthorized origin: {origin}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
    
    logger.info(f"✅ WebSocket connection authorized from {origin}")

async def verify_websocket_auth(websocket: WebSocket) -> None:
    """Verify WebSocket authentication if API key is required."""
    if not security_config.api_key:
        return  # No auth required
    
    # Check for API key in query parameters or headers
    api_key = websocket.query_params.get("api_key") or websocket.headers.get("x-api-key")
    
    if not api_key or api_key != security_config.api_key:
        logger.warning(f"WebSocket connection with invalid API key from {websocket.client}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

def setup_security(app: FastAPI) -> None:
    """Configure all security features for MISO FastAPI app."""
    logger.info("🔒 Initializing MISO security configuration...")
    
    # Add security headers middleware
    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # CSP for dashboard protection
        if "text/html" in response.headers.get("content-type", ""):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.plot.ly; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' ws: wss:;"
            )
        
        return response
    
    # Configure CORS and rate limiting  
    add_cors_middleware(app)
    add_rate_limiting(app)
    
    logger.info("🔒 Security configuration completed")

# WebSocket security decorator
def secure_websocket(websocket_handler):
    """Decorator for securing WebSocket endpoints."""
    async def wrapper(websocket: WebSocket, *args, **kwargs):
        # Verify origin and authentication before accepting connection
        await verify_websocket_origin(websocket)
        await verify_websocket_auth(websocket)
        
        # Accept connection after security checks
        await websocket.accept()
        
        try:
            return await websocket_handler(websocket, *args, **kwargs)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
    
    return wrapper

# Health check bypass for monitoring
def is_health_check(request) -> bool:
    """Check if request is a health check that should bypass auth."""
    return request.url.path in ["/health", "/metrics", "/ready"]
