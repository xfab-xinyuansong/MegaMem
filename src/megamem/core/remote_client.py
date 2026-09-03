"""HTTP adapter for a deployed MegaMem service."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import requests


class RemoteMemoryError(RuntimeError):
    """Raised when a remote memory request cannot be completed."""


class RemoteMemoryClient:
    """Small authenticated client for remote add and query operations."""

    def __init__(
        self,
        api_key: str,
        server_url: Optional[str] = None,
        timeout_seconds: float = 30.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for remote mode")
        configured_url = (
            server_url
            or os.getenv("MEGAMEM_SERVER_URL")
            or "http://localhost:8000"
        )
        self.server_url = configured_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout_seconds: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        url = f"{self.server_url}{path}"
        try:
            response = self._session.request(
                method,
                url,
                timeout=timeout_seconds or self.timeout_seconds,
                **kwargs,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            raise RemoteMemoryError(f"Remote memory request failed: {method} {url}") from exc
        except ValueError as exc:
            raise RemoteMemoryError(f"Remote memory service returned invalid JSON: {url}") from exc

    def add(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Persist memory content through the remote service."""
        return self._request(
            "POST",
            "/api/v1/memory/add",
            json={"context": context, "metadata": metadata or {}},
        )

    def query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        enable_hybrid_search: bool = False,
        enable_llm_filter: bool = False,
        query_mode: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Search memories through the remote service."""
        mode = getattr(query_mode, "name", query_mode)
        payload = {
            "context": context,
            "top_k": top_k,
            "where": where,
            "include": include,
            "enable_hybrid_search": enable_hybrid_search,
            "enable_llm_filter": enable_llm_filter,
            "query_mode": mode,
            **kwargs,
        }
        return self._request(
            "POST",
            "/api/v1/memory/query",
            json={key: value for key, value in payload.items() if value is not None},
        )

    def planner_query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        latency_tracker: Any = None,
    ) -> Any:
        """Run planner retrieval through the remote service."""
        del latency_tracker
        query = context if isinstance(context, str) else str(context)
        return self._request(
            "GET",
            "/api/memory/planner_query",
            timeout_seconds=60.0,
            params={"q": query, "k": top_k},
        )
