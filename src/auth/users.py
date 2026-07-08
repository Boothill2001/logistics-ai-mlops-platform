"""User store — loads the simulated user directory.

Demo stand-in for an identity provider (OIDC/LDAP). The API trusts the
X-User-Id header; production would validate a signed token instead.
"""
import json
from dataclasses import dataclass
from functools import lru_cache

from src.config import settings


@dataclass(frozen=True)
class User:
    user_id: str
    role: str
    customer_access: tuple[str, ...]
    permission_groups: tuple[str, ...]


@lru_cache(maxsize=1)
def _load_users() -> dict[str, User]:
    raw = json.loads((settings.data_dir / "users.json").read_text(encoding="utf-8"))
    return {
        u["user_id"]: User(
            user_id=u["user_id"],
            role=u["role"],
            customer_access=tuple(u["customer_access"]),
            permission_groups=tuple(u["permission_groups"]),
        )
        for u in raw
    }


def get_user(user_id: str) -> User | None:
    return _load_users().get(user_id)
