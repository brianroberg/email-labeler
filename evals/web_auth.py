"""Simple shared-secret authentication for eval web UI."""

import os
import secrets

from fastapi import Request
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

SECRET = os.environ.get("EVAL_WEB_SECRET", "")


def is_authenticated(request: Request) -> bool:
    """Check if a request carries valid credentials (cookie or header)."""
    if not SECRET:
        return True  # auth disabled when no secret configured
    cookie = request.cookies.get("eval_session", "")
    if cookie and secrets.compare_digest(cookie, SECRET):
        return True
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and secrets.compare_digest(auth[7:], SECRET):
        return True
    return False


class AuthMiddleware(BaseHTTPMiddleware):
    """Redirect unauthenticated requests to /login (skip /login and /static)."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path == "/login" or path.startswith("/static"):
            return await call_next(request)
        if not is_authenticated(request):
            return RedirectResponse(url="/login", status_code=302)
        return await call_next(request)
