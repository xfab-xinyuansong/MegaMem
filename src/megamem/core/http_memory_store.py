"""
Memory store that proxies operations to a remote HTTP service.
"""

import json
import requests
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

from omegaconf import DictConfig
from chromadb.api.types import Where

from megamem.core.base import MemoryBase
from megamem.utils.misc import index_to_id


class HttpMemoryStore(MemoryBase):
    """
    HTTP-backed memory store implementation.

    All memory primitives are forwarded to a remote service over HTTP.
    """

    def __init__(self, cfg: DictConfig):
        """
        Set up the HTTP memory store.

        Args:
            cfg: Configuration object containing HTTP memory settings
        """
        self.cfg = cfg

        # Pull HTTP options from the supplied config object.
        self.base_url = cfg.memory.http.base_url.rstrip('/')
        self.timeout = cfg.memory.get('http', {}).get('timeout', 30)
        self.headers = {
            'Content-Type': 'application/json',
        }

        # Layer in authentication when configured.
        if hasattr(cfg.memory.http, 'api_key'):
            self.headers['Authorization'] = f"Bearer {cfg.memory.http.api_key}"
        elif hasattr(cfg.memory.http, 'auth_header'):
            auth_config = cfg.memory.http.auth_header
            self.headers[auth_config.name] = auth_config.value

        # Remember which collection the API calls should target.
        self.collection_name = cfg.memory.collection_name

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Issue an HTTP request to the remote memory service.

        Args:
            method: HTTP verb (GET, POST, PUT, DELETE)
            endpoint: API path
            data: Request payload

        Returns:
            Response data as a dict

        Raises:
            Exception: If the request fails
        """
        url = urljoin(self.base_url, endpoint)

        try:
            verb = method.upper()
            if verb == 'GET':
                response = requests.get(url, headers=self.headers, params=data, timeout=self.timeout)
            elif verb == 'POST':
                response = requests.post(url, headers=self.headers, json=data, timeout=self.timeout)
            elif verb == 'PUT':
                response = requests.put(url, headers=self.headers, json=data, timeout=self.timeout)
            elif verb == 'DELETE':
                response = requests.delete(url, headers=self.headers, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()

            # Treat empty bodies (or 204 No Content) as an empty dict.
            if response.status_code == 204 or not response.content:
                return {}

            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"HTTP request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse response JSON: {str(e)}")

    def upsert(
        self,
        key: str,
        value: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Insert or update the memory record keyed by 'key'.

        Args:
            key: Natural-language key to be embedded
            value: Value/content to store
            extra_meta: Additional metadata to store

        Returns:
            Record ID (derived from key)
        """
        rid = index_to_id(key)

        # Assemble the metadata payload.
        meta = {"original_key": key, "value": value}
        if extra_meta:
            meta = {**meta, **extra_meta}

        # Compose the request body.
        data = {
            "collection_name": self.collection_name,
            "id": rid,
            "key": key,
            "value": value,
            "metadata": meta
        }

        # Hit the upsert endpoint.
        endpoint = "/api/memory/upsert"
        self._make_request("POST", endpoint, data)

        return rid

    def query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        k: int = 5,
        where: Optional[Where] = None,
        include: Optional[List[str]] = None,
    ):
        """
        Vector search to surface memories similar to the given context.

        Args:
            context: Context to search for. Can be:
                - str: Natural language query text
                - List[str]: Multiple query strings
                - List[Dict[str, str]]: Structured context with key-value pairs
            k: Number of results to return
            where: Filter conditions for metadata-based filtering
            include: Fields to include in results (e.g., ["metadatas", "distances"])

        Returns:
            Query result dict matching the ChromaDB shape
        """
        include = include or ["metadatas", "distances"]

        # Coerce the various accepted context shapes into a single query string.
        if isinstance(context, str):
            query_text = context
        elif isinstance(context, list):
            if all(isinstance(item, str) for item in context):
                # List of strings -> concatenate them.
                query_text = " ".join(context)
            elif all(isinstance(item, dict) for item in context):
                # List of dicts -> pull values out and concatenate.
                query_text = " ".join(
                    " ".join(item.values()) for item in context
                )
            else:
                raise ValueError("Context list must contain either all strings or all dictionaries")
        else:
            raise ValueError("Context must be a string, list of strings, or list of dictionaries")

        # Build the request payload.
        data = {
            "collection_name": self.collection_name,
            "query_text": query_text,
            "n_results": k,
            "include": include
        }

        if where:
            data["where"] = where

        # Send the query request.
        endpoint = "/api/memory/query"
        return self._make_request("POST", endpoint, data)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single record using its natural-language key.

        Args:
            key: Natural-language key to look up

        Returns:
            Dict containing id, metadata, document — or None if not found
        """
        record_id = index_to_id(key)

        # Build the request payload.
        data = {
            "collection_name": self.collection_name,
            "id": record_id
        }

        # Issue the get request.
        endpoint = "/api/memory/get"
        try:
            result = self._make_request("GET", endpoint, data)

            # Handle a missing record.
            if not result or not result.get("found", True):
                return None

            return {
                "id": result.get("id"),
                "metadata": result.get("metadata"),
                "document": result.get("document"),
            }

        except Exception as e:
            # Map 404 / not-found responses to None.
            if "404" in str(e) or "not found" in str(e).lower():
                return None
            raise

    def delete(self, key: str) -> None:
        """
        Remove the record matching the given natural-language key.

        Args:
            key: Natural-language key to delete
        """
        record_id = index_to_id(key)

        # Build the request payload.
        data = {
            "collection_name": self.collection_name,
            "id": record_id
        }

        # Hit the delete endpoint.
        endpoint = "/api/memory/delete"
        self._make_request("DELETE", endpoint, data)

    def list_memories(self, limit: int = 10) -> Dict[str, Any]:
        """
        Enumerate memory records in the collection.

        Args:
            limit: Max number of records to return

        Returns:
            Dict containing memory records
        """
        # Build the request payload.
        data = {
            "collection_name": self.collection_name,
            "limit": limit,
            "offset": 0,
            "include": ["documents", "metadatas"]
        }

        # Hit the list endpoint.
        endpoint = "/api/memory/list"
        return self._make_request("GET", endpoint, data)

    def count(self) -> int:
        """
        Number of records currently stored in the collection.

        Returns:
            Integer count
        """
        # Build the request payload.
        data = {
            "collection_name": self.collection_name
        }

        # Hit the count endpoint.
        endpoint = "/api/memory/count"
        result = self._make_request("GET", endpoint, data)

        return result.get("count", 0)
