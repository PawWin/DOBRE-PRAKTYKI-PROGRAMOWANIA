import base64
import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta
from typing import Iterable, Sequence
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "super-secret-key")
ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password


def _urlsafe_b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("utf-8")


def _urlsafe_b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _sign(value: str) -> str:
    digest = hmac.new(SECRET_KEY.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).digest()
    return _urlsafe_b64encode(digest)


class TokenValidationError(Exception):
    pass


def create_access_token(*, subject: str, roles: Sequence[str], expires_delta: timedelta | None = None) -> str:
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {
        "sub": subject,
        "roles": list(roles),
        "exp": int(expire.timestamp()),
    }
    header = {"alg": ALGORITHM, "typ": "JWT"}
    header_segment = _urlsafe_b64encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    payload_segment = _urlsafe_b64encode(json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8"))
    signing_input = f"{header_segment}.{payload_segment}"
    signature_segment = _sign(signing_input)
    return f"{signing_input}.{signature_segment}"


def decode_token(token: str) -> dict:
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
    except ValueError as exc:
        raise TokenValidationError("Invalid token format") from exc

    signing_input = f"{header_segment}.{payload_segment}"
    expected_signature = _sign(signing_input)
    if not hmac.compare_digest(expected_signature, signature_segment):
        raise TokenValidationError("Invalid token signature")

    payload = json.loads(_urlsafe_b64decode(payload_segment))
    exp = payload.get("exp")
    if exp is not None:
        if datetime.utcnow().timestamp() > float(exp):
            raise TokenValidationError("Token expired")
    return payload


def parse_roles(raw_roles: str | Iterable[str] | None) -> list[str]:
    if raw_roles is None:
        return []
    if isinstance(raw_roles, str):
        return [role.strip() for role in raw_roles.split(",") if role.strip()]
    return [str(role).strip() for role in raw_roles if str(role).strip()]


def format_roles(roles: Sequence[str]) -> str:
    normalized = parse_roles(roles)
    return ",".join(dict.fromkeys(normalized)) or "ROLE_USER"
