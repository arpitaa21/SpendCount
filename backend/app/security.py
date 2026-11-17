import os, hmac, hashlib, base64, json, time
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from .database import get_db
from .models import User

SECRET = os.getenv("SECRET_KEY", "local-development-secret-change-me").encode()
bearer = HTTPBearer(auto_error=False)

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 240000)
    return base64.urlsafe_b64encode(salt + digest).decode()

def verify_password(password: str, encoded: str) -> bool:
    try:
        raw = base64.urlsafe_b64decode(encoded.encode())
        return hmac.compare_digest(raw[16:], hashlib.pbkdf2_hmac("sha256", password.encode(), raw[:16], 240000))
    except Exception:
        return False

def create_token(user_id: int) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"sub": user_id, "exp": int(time.time()) + int(os.getenv("ACCESS_TOKEN_MINUTES", "10080")) * 60}).encode()).decode().rstrip("=")
    sig = base64.urlsafe_b64encode(hmac.new(SECRET, payload.encode(), hashlib.sha256).digest()).decode().rstrip("=")
    return payload + "." + sig

def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer), db: Session = Depends(get_db)) -> User:
    if not credentials:
        raise HTTPException(401, "Sign in to continue")
    try:
        payload, sig = credentials.credentials.split(".")
        expected = base64.urlsafe_b64encode(hmac.new(SECRET, payload.encode(), hashlib.sha256).digest()).decode().rstrip("=")
        if not hmac.compare_digest(sig, expected): raise ValueError()
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if data["exp"] < time.time(): raise ValueError()
        user = db.get(User, int(data["sub"]))
        if not user: raise ValueError()
        return user
    except Exception:
        raise HTTPException(401, "Session expired. Please sign in again")
