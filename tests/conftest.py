import sys
from itertools import count
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.api import app
from app.db import get_db
from app.models import Base, User
from app.security import format_roles, hash_password

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def db_session():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def user_factory(db_session):
    sequence = count(1)

    def _create_user(*, username=None, email=None, password="pass123", roles=None):
        idx = next(sequence)
        user = User(
            username=username or f"user{idx}",
            email=email or f"user{idx}@example.com",
            hashed_password=hash_password(password),
            roles=format_roles(roles or ["ROLE_USER"]),
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        return user, password

    return _create_user


@pytest.fixture()
def admin_credentials(user_factory):
    user, password = user_factory(
        username="admin",
        email="admin@example.com",
        password="adminpass123",
        roles=["ROLE_ADMIN", "ROLE_USER"],
    )
    return {"username": user.username, "password": password}


@pytest.fixture()
def client(db_session):
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def authorized_client(client, admin_credentials):
    response = client.post("/login", json=admin_credentials)
    assert response.status_code == 200
    token = response.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client
