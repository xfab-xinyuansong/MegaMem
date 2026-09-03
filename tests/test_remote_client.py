from __future__ import annotations

from typing import Any

from megamem.core.remote_client import RemoteMemoryClient


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, bool]:
        return {"ok": True}


class _Session:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        self.calls.append((method, url, kwargs))
        return _Response()


def test_remote_client_normalizes_url_and_authenticates() -> None:
    session = _Session()
    client = RemoteMemoryClient(
        api_key="test-token",
        server_url="https://memory.example/",
        session=session,
    )

    assert client.add("policy text", metadata={"source": "policy"}) == {"ok": True}
    assert session.headers["Authorization"] == "Bearer test-token"
    assert session.calls[0][0:2] == (
        "POST",
        "https://memory.example/api/v1/memory/add",
    )


def test_remote_query_omits_unset_optional_fields() -> None:
    session = _Session()
    client = RemoteMemoryClient("test-token", session=session)

    client.query("deadline", top_k=3)
    payload = session.calls[0][2]["json"]

    assert payload["top_k"] == 3
    assert "where" not in payload
    assert "query_mode" not in payload
