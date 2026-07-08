"""RBAC policy checks.

Deny-by-default: anything not explicitly granted is refused, and refusals
return a *controlled* error message that confirms nothing about whether the
requested resource exists (no "document X is restricted" — that itself leaks).
"""
from src.auth.users import User

DENIED_MESSAGE = (
    "You don't have permission to access this information. "
    "Contact your administrator if you believe this is an error."
)


def can_access_customer(user: User, customer_id: str) -> bool:
    return customer_id in user.customer_access


def allowed_permission_groups(user: User) -> list[str]:
    """Groups usable as a retrieval filter — the ONLY groups whose documents
    may ever reach the LLM context for this user."""
    return list(user.permission_groups)
