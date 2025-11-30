from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db import get_db
from app.endpoints.auth import get_current_user, get_token_payload, require_admin
from app.models import User
from app.schemas import UserCreate
from app.security import format_roles, hash_password, parse_roles

router = APIRouter(tags=["users"])


@router.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(user_in: UserCreate, db: Session = Depends(get_db), _: User = Depends(require_admin)):
    existing = (
        db.query(User)
        .filter(or_(User.username == user_in.username, User.email == user_in.email))
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this username or email already exists",
        )
    db_user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
        roles=format_roles(user_in.roles),
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {
        "id": db_user.id,
        "username": db_user.username,
        "email": db_user.email,
        "roles": parse_roles(db_user.roles),
    }


@router.get("/user_details")
def user_details(
    payload: dict = Depends(get_token_payload), _: User = Depends(get_current_user)
):
    return {
        "sub": payload.get("sub"),
        "roles": payload.get("roles", []),
        "exp": payload.get("exp"),
    }
