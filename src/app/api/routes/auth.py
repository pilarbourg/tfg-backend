import os
import secrets
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(request: LoginRequest):
    """
    Validates admin credentials against environment variables.

    Returns a success response if credentials match, 401 otherwise.
    """
    expected_user = os.getenv("ADMIN_USERNAME")
    expected_pass = os.getenv("ADMIN_PASSWORD")

    if not expected_user or not expected_pass:
        raise HTTPException(status_code=500, detail="Server credentials not configured.")

    if not secrets.compare_digest(request.username, expected_user) or \
       not secrets.compare_digest(request.password, expected_pass):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    return {"success": True}