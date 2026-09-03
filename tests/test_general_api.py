from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from megamem import GeneralAPIClient
from megamem.core.general_api import GeneralAPIError


class _Response:
    def __init__(self, payload: Any, status_code: int = 200, text: str = "") -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = text

    def json(self) -> Any:
        return self.payload


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def test_general_chat_contract() -> None:
    session = _Session(
        [
            _Response(
                {
                    "choices": [{"message": {"content": "grounded answer"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                }
            )
        ]
    )
    client = GeneralAPIClient(
        base_url="https://gateway.example/v1/",
        api_key="test-secret-value",
        session=session,
        max_retries=0,
    )

    response = client.chat.completions.create(
        model="chat-model",
        messages=[{"role": "user", "content": "question"}],
        max_tokens=64,
    )

    assert session.calls[0]["url"] == "https://gateway.example/v1/chat/completions"
    assert session.calls[0]["headers"]["Authorization"] == "Bearer test-secret-value"
    assert response.choices[0].message.content == "grounded answer"
    assert response.usage.total_tokens == 16
    assert response.provider == "general"


def test_general_embedding_contract_orders_vectors() -> None:
    session = _Session(
        [
            _Response(
                {
                    "data": [
                        {"index": 1, "embedding": [0, 1]},
                        {"index": 0, "embedding": [1, 0]},
                    ]
                }
            )
        ]
    )
    client = GeneralAPIClient(
        base_url="https://gateway.example/v1",
        api_key="test-secret-value",
        session=session,
        max_retries=0,
    )

    response = client.embeddings.create(model="embed-model", input=["a", "b"])

    assert [item.embedding for item in response.data] == [[1.0, 0.0], [0.0, 1.0]]
    assert response.provider == "general"


def test_general_structured_response_is_parsed_locally() -> None:
    class Result(BaseModel):
        answer: str

    session = _Session(
        [_Response({"choices": [{"message": {"content": '{"answer":"ok"}'}}]})]
    )
    client = GeneralAPIClient(
        base_url="https://gateway.example/v1",
        api_key="test-secret-value",
        session=session,
        max_retries=0,
    )

    response = client.chat.completions.parse(
        model="chat-model",
        messages=[{"role": "user", "content": "question"}],
        response_format=Result,
    )

    assert response.choices[0].message.parsed == Result(answer="ok")
    assert session.calls[0]["json"]["response_format"] == {"type": "json_object"}


def test_general_error_redacts_api_key() -> None:
    secret = "test-secret-value"
    session = _Session([_Response({"error": {"message": secret}}, status_code=400)])
    client = GeneralAPIClient(
        base_url="https://gateway.example/v1",
        api_key=secret,
        session=session,
        max_retries=0,
    )

    with pytest.raises(GeneralAPIError) as exc:
        client.chat.completions.create(model="chat-model", messages=[])

    assert secret not in str(exc.value)
    assert "REDACTED" in str(exc.value)
