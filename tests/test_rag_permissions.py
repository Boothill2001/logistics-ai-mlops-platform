"""Phase 2 tests: permission-aware retrieval and leakage control.

These are the security-critical tests: they assert what each user CANNOT see.
"""
import pytest

from src.auth.users import get_user
from src.copilot.llm import MockLLM
from src.rag.answerer import NO_DATA_MESSAGE, answer_question
from src.rag.store import DocStore


@pytest.fixture(scope="module")
def store():
    return DocStore()


def _doc_ids(chunks):
    return {c.document_id for c in chunks}


def test_sales_can_see_customer_a_contract(store):
    sales = get_user("sales_001")
    chunks = store.retrieve("delay compensation contract customer A rebate", sales)
    assert "contract_customer_A" in _doc_ids(chunks)


def test_intern_cannot_see_customer_a_contract(store):
    intern = get_user("intern_001")
    chunks = store.retrieve("delay compensation contract customer A rebate", intern)
    ids = _doc_ids(chunks)
    assert "contract_customer_A" not in ids
    assert "invoice_customer_A" not in ids
    # but public docs are still reachable
    assert ids <= {"shipping_policy", "delay_sop"}


@pytest.mark.parametrize("user_id", ["sales_001", "ops_001", "intern_001"])
def test_private_note_hidden_from_everyone_but_admin(store, user_id):
    user = get_user(user_id)
    # Query crafted to match the private note strongly
    chunks = store.retrieve("private note RFQ floor price negotiation customer A", user)
    assert "dangerous_private_note" not in _doc_ids(chunks)


def test_admin_can_see_private_note(store):
    admin = get_user("admin_001")
    chunks = store.retrieve("private note RFQ floor price negotiation customer A", admin)
    assert "dangerous_private_note" in _doc_ids(chunks)


def test_answer_has_citations_and_no_data_fallback(store):
    llm = MockLLM()
    sales = get_user("sales_001")
    answer = answer_question("What is the delay compensation policy for customer A?",
                             sales, store, llm)
    assert answer.grounded
    assert answer.sources  # citations present

    intern = get_user("intern_001")
    blocked = answer_question("Is there any private note about customer A RFQ floor price?",
                              intern, store, llm)
    # Intern gets either the controlled no-data message, or an answer grounded
    # ONLY in public docs — never the private note.
    assert "dangerous_private_note" not in blocked.sources
    if not blocked.grounded:
        assert blocked.text == NO_DATA_MESSAGE
    assert "2,600" not in blocked.text  # the confidential floor price never leaks
