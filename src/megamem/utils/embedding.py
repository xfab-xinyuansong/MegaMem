from __future__ import annotations

import logging
import os
from typing import List, Optional

from omegaconf import DictConfig

from megamem.core.general_api import GeneralAPIClient

logger = logging.getLogger(__name__)

_LOCAL_MODEL_CACHE: dict = {}


def _cfg_get(cfg: Optional[DictConfig], key: str, default: str = "") -> str:
    if cfg is None:
        return default
    block = getattr(cfg, "embedding", None)
    if block is None:
        return default
    return getattr(block, key, default)


def get_general_embedding_client(cfg: Optional[DictConfig] = None) -> GeneralAPIClient:
    """Build a general embeddings API client."""
    base_url = (
        os.getenv("EMBEDDING_API_BASE")
        or os.getenv("LLM_API_BASE")
        or _cfg_get(cfg, "embedding_api_base")
    )
    api_key = (
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("LLM_API_KEY")
        or _cfg_get(cfg, "embedding_api_key")
    )
    if not base_url or not api_key:
        raise RuntimeError(
            "Embedding API base/key not configured. "
            "Set EMBEDDING_API_BASE and EMBEDDING_API_KEY, or enable local embeddings."
        )
    return GeneralAPIClient(base_url=base_url, api_key=api_key)


class BaseEmbeddingModel:
    """Embedding wrapper with a local-first default.

    Local sentence-transformers embeddings are used by default. Set
    ``MEGAMEM_LOCAL_EMBEDDING=0`` to route through a hosted endpoint
    configured with general `EMBEDDING_API_*` variables.
    """

    def __init__(
        self,
        cfg: DictConfig,
        client: Optional[GeneralAPIClient] = None,
    ):
        self.cfg = cfg

        local_flag = os.getenv("MEGAMEM_LOCAL_EMBEDDING", "1").lower()
        if local_flag not in ("0", "false", "no"):
            model_name = os.getenv(
                "MEGAMEM_LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            )
            cached = _LOCAL_MODEL_CACHE.get(model_name)
            if cached is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError(
                        "MEGAMEM_LOCAL_EMBEDDING=1 but sentence-transformers is "
                        "not installed. Install: pip install sentence-transformers"
                    ) from exc
                logger.warning(
                    "Loading local sentence-transformers embedding model "
                    f"({model_name}). Record this choice with the run metadata."
                )
                cached = SentenceTransformer(model_name)
                _LOCAL_MODEL_CACHE[model_name] = cached
            self.client = None
            self._local_model = cached
            self._is_local = True
            self._local_model_name = model_name
            return

        self._is_local = False
        self._local_model = None
        self.client = client if client else get_general_embedding_client(cfg)

    def get_client(self) -> Optional[GeneralAPIClient]:
        """Return the hosted embedding client, or None in local mode."""
        return self.client

    def generate_embeddings(
        self,
        input: List[str],
    ) -> List[List[float]]:
        """Embed a batch of strings."""
        if getattr(self, "_is_local", False):
            vecs = self._local_model.encode(
                input,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return [vec.tolist() for vec in vecs]

        model_name = (
            os.getenv("EMBEDDING_MODEL")
            or _cfg_get(self.cfg, "model")
            or "sentence-transformers/all-MiniLM-L6-v2"
        )
        raw = self.client.embeddings.create(input=input, model=model_name).data
        return [item.embedding for item in raw]
