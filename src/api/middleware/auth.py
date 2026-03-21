from fastapi import Header, HTTPException


async def authenticate_request(authorization: str | None = Header(default=None)) -> bool:
    # Local/dev mode: allow requests without token; if token is provided, require Bearer format.
    if authorization is None:
        return True

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    return True
