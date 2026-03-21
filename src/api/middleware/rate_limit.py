import time

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 120):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._hits: dict[str, list[float]] = {}

    async def dispatch(self, request, call_next):
        client = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60
        hits = self._hits.get(client, [])
        hits = [t for t in hits if t >= window_start]

        if len(hits) >= self.requests_per_minute:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        hits.append(now)
        self._hits[client] = hits
        return await call_next(request)
