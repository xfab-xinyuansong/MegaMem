"""General JSON model-gateway client used by MegaMem.

The client implements the small chat and embedding surface MegaMem needs
without binding the package to a cloud vendor or vendor SDK. Endpoints,
models, authentication, and routing remain deployment configuration.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter


class GeneralAPIError(RuntimeError):
    """Raised when a general model-gateway request cannot be completed."""


class GeneralBadRequestError(GeneralAPIError):
    """Raised for a non-retryable 4xx gateway response."""


class GeneralContentFilterError(GeneralAPIError):
    """Raised when the gateway reports a filtered completion."""


@dataclass(frozen=True)
class GeneralUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class GeneralMessage:
    content: str
    parsed: Any = None


@dataclass(frozen=True)
class GeneralChoice:
    message: GeneralMessage
    finish_reason: str = ""


@dataclass(frozen=True)
class GeneralChatResponse:
    choices: List[GeneralChoice]
    usage: GeneralUsage
    raw: Dict[str, Any]
    provider: str = "general"


@dataclass(frozen=True)
class GeneralEmbedding:
    embedding: List[float]
    index: int


@dataclass(frozen=True)
class GeneralEmbeddingResponse:
    data: List[GeneralEmbedding]
    usage: GeneralUsage
    raw: Dict[str, Any]
    provider: str = "general"


def build_general_session(pool_max: int = 256, pool_connections: int = 128):
    """Create a requests session with bounded connection pools."""
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_max,
        max_retries=0,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _usage(raw: Dict[str, Any]) -> GeneralUsage:
    usage = raw.get("usage") or {}
    return GeneralUsage(
        prompt_tokens=int(
            usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        ),
        completion_tokens=int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        ),
    )


def _parse_structured(content: str, response_format: Any) -> Any:
    if hasattr(response_format, "model_validate_json"):
        return response_format.model_validate_json(content)
    if hasattr(response_format, "parse_raw"):
        return response_format.parse_raw(content)
    data = json.loads(content)
    return response_format(**data)


class _ChatCompletions:
    def __init__(self, client: "GeneralAPIClient"):
        self._client = client

    def create(self, **payload: Any) -> GeneralChatResponse:
        return self._client._chat_completion(payload)

    def parse(self, *, response_format: Any, **payload: Any) -> GeneralChatResponse:
        payload["response_format"] = {"type": "json_object"}
        response = self._client._chat_completion(payload)
        content = response.choices[0].message.content
        parsed = _parse_structured(content, response_format)
        choice = response.choices[0]
        return GeneralChatResponse(
            choices=[
                GeneralChoice(
                    message=GeneralMessage(content=content, parsed=parsed),
                    finish_reason=choice.finish_reason,
                )
            ],
            usage=response.usage,
            raw=response.raw,
        )


class _Chat:
    def __init__(self, client: "GeneralAPIClient"):
        self.completions = _ChatCompletions(client)


class _Embeddings:
    def __init__(self, client: "GeneralAPIClient"):
        self._client = client

    def create(self, **payload: Any) -> GeneralEmbeddingResponse:
        return self._client._embedding(payload)


class GeneralAPIClient:
    """Client for the MegaMem general chat and embedding contract."""

    provider = "general"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        max_retries: int = 2,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not base_url or not api_key:
            raise ValueError("base_url and api_key are required")
        if timeout <= 0 or max_retries < 0:
            raise ValueError("timeout must be positive and max_retries non-negative")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = session or build_general_session()
        self.chat = _Chat(self)
        self.embeddings = _Embeddings(self)

    def _endpoint(self, resource: str) -> str:
        suffix = f"/{resource.lstrip('/')}"
        return self.base_url if self.base_url.endswith(suffix) else self.base_url + suffix

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _redact(self, text: str) -> str:
        rendered = text.replace(self._api_key, "<REDACTED>")
        return re.sub(
            r"(Bearer\s+)[A-Za-z0-9._+/=-]{12,}",
            r"\1<REDACTED>",
            rendered,
            flags=re.IGNORECASE,
        )

    def _post(self, resource: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[GeneralAPIError] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.post(
                    self._endpoint(resource),
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = GeneralAPIError(self._redact(str(exc)))
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise last_error from exc

            try:
                raw = response.json()
            except ValueError as exc:
                raw = {}
                detail = self._redact(response.text[:300])
                retryable = (
                    response.status_code in {408, 409, 429}
                    or response.status_code >= 500
                )
                error_type = (
                    GeneralAPIError if retryable else GeneralBadRequestError
                )
                error = error_type(
                    f"General API returned non-JSON HTTP {response.status_code}: {detail}"
                )
                if retryable and attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise error from exc

            if not isinstance(raw, dict):
                raise GeneralAPIError("General API response must be a JSON object")
            if response.status_code < 400:
                return raw

            detail = self._redact(json.dumps(raw, ensure_ascii=True)[:300])
            message = f"General API HTTP {response.status_code}: {detail}"
            retryable = response.status_code in {408, 409, 429} or response.status_code >= 500
            error_type = GeneralAPIError if retryable else GeneralBadRequestError
            last_error = error_type(message)
            if retryable and attempt < self.max_retries:
                time.sleep(min(2**attempt, 8))
                continue
            raise last_error

        raise last_error or GeneralAPIError("General API request failed")

    def _chat_completion(self, payload: Dict[str, Any]) -> GeneralChatResponse:
        raw = self._post("chat/completions", payload)
        choices = raw.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise GeneralAPIError("General chat response is missing choices")
        first = choices[0]
        finish_reason = str(first.get("finish_reason") or "")
        if finish_reason == "content_filter":
            raise GeneralContentFilterError("General API filtered the completion")
        message = first.get("message") or {}
        content = _content_text(message.get("content"))
        return GeneralChatResponse(
            choices=[
                GeneralChoice(
                    message=GeneralMessage(content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=_usage(raw),
            raw=raw,
        )

    def _embedding(self, payload: Dict[str, Any]) -> GeneralEmbeddingResponse:
        raw = self._post("embeddings", payload)
        data = raw.get("data") or []
        if not isinstance(data, list):
            raise GeneralAPIError("General embedding response has invalid data")
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        embeddings = []
        for item in ordered:
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise GeneralAPIError("General embedding response has invalid vector data")
            embeddings.append(
                GeneralEmbedding(
                    embedding=[float(value) for value in vector],
                    index=int(item.get("index", len(embeddings))),
                )
            )
        return GeneralEmbeddingResponse(data=embeddings, usage=_usage(raw), raw=raw)
