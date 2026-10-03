import json
import requests
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

from omegaconf import DictConfig
from chromadb.api.types import Where

from megamem.core.base import MemoryBase
from megamem.utils.misc import index_to_id


class HttpMemoryStore(MemoryBase):

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.base_url = cfg.memory.http.base_url.rstrip('/')
        self.timeout = cfg.memory.get('http', {}).get('timeout', 30)
        self.headers = {
            'Content-Type': 'application/json',
        }

        if hasattr(cfg.memory.http, 'api_key'):
            self.headers['Authorization'] = f"Bearer {cfg.memory.http.api_key}"
        elif hasattr(cfg.memory.http, 'auth_header'):
            auth_config = cfg.memory.http.auth_header
            self.headers[auth_config.name] = auth_config.value

        self.collection_name = cfg.memory.collection_name

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
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
        rid = index_to_id(key)

        meta = {"original_key": key, "value": value}
        if extra_meta:
            meta = {**meta, **extra_meta}

        data = {
            "collection_name": self.collection_name,
            "id": rid,
            "key": key,
            "value": value,
            "metadata": meta
        }

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
        include = include or ["metadatas", "distances"]

        if isinstance(context, str):
            query_text = context
        elif isinstance(context, list):
            if all(isinstance(item, str) for item in context):
                query_text = " ".join(context)
            elif all(isinstance(item, dict) for item in context):
                query_text = " ".join(
                    " ".join(item.values()) for item in context
                )
            else:
                raise ValueError("Context list must contain either all strings or all dictionaries")
        else:
            raise ValueError("Context must be a string, list of strings, or list of dictionaries")

        data = {
            "collection_name": self.collection_name,
            "query_text": query_text,
            "n_results": k,
            "include": include
        }

        if where:
            data["where"] = where

        endpoint = "/api/memory/query"
        return self._make_request("POST", endpoint, data)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        record_id = index_to_id(key)

        data = {
            "collection_name": self.collection_name,
            "id": record_id
        }

        endpoint = "/api/memory/get"
        try:
            result = self._make_request("GET", endpoint, data)

            if not result or not result.get("found", True):
                return None

            return {
                "id": result.get("id"),
                "metadata": result.get("metadata"),
                "document": result.get("document"),
            }

        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return None
            raise

    def delete(self, key: str) -> None:
        record_id = index_to_id(key)

        data = {
            "collection_name": self.collection_name,
            "id": record_id
        }

        endpoint = "/api/memory/delete"
        self._make_request("DELETE", endpoint, data)

    def list_memories(self, limit: int = 10) -> Dict[str, Any]:
        data = {
            "collection_name": self.collection_name,
            "limit": limit,
            "offset": 0,
            "include": ["documents", "metadatas"]
        }

        endpoint = "/api/memory/list"
        return self._make_request("GET", endpoint, data)

    def count(self) -> int:
        data = {
            "collection_name": self.collection_name
        }

        endpoint = "/api/memory/count"
        result = self._make_request("GET", endpoint, data)

        return result.get("count", 0)
