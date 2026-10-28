"""Security utilities: JWT token generation and authentication."""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt

from app.core.config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# In-memory test accounts
TEST_USERS = {
    settings.TEST_ADMIN_USERNAME: {
        "username": settings.TEST_ADMIN_USERNAME,
        "password": settings.TEST_ADMIN_PASSWORD,
        "role": "admin",
        "description": "Administrator account with full access",
    },
    settings.TEST_USER_USERNAME: {
        "username": settings.TEST_USER_USERNAME,
        "password": settings.TEST_USER_PASSWORD,
        "role": "tester",
        "description": "Standard metrology tester account",
    },
}


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = TEST_USERS.get(username)
    if not user:
        return None
    if user["password"] != password:
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": now})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or token expired",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except (jwt.PyJWTError, Exception):
        raise credentials_exception

    user = TEST_USERS.get(username)
    if user is None:
        raise credentials_exception
    return user
