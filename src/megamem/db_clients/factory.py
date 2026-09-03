"""Configuration-driven factory for vector database clients."""
from omegaconf import DictConfig

from megamem.db_clients.base import VectorDBClient
from megamem.db_clients.chromadb_client import ChromaDBClient
from megamem.db_clients.redis_client import RedisVectorDBClient


def create_vector_db_client(cfg: DictConfig) -> VectorDBClient:
    """Build a concrete ``VectorDBClient`` from configuration.

    The selected backend is taken from ``cfg.memory.db_type`` (case
    insensitive); ``"chromadb"``/``"chroma"`` and ``"redis"`` are
    supported. ChromaDB is the default when the option is missing.

    Args:
        cfg: full configuration object.

    Returns:
        A configured ``VectorDBClient`` subclass instance.

    Raises:
        ValueError: when ``db_type`` is not one of the supported values.
    """
    db_type = cfg.memory.get("db_type", "chromadb").lower()

    if db_type in ("chromadb", "chroma"):
        return ChromaDBClient(cfg)
    if db_type == "redis":
        return RedisVectorDBClient(cfg)

    raise ValueError(
        f"Unsupported database type: {db_type}. "
        f"Supported types: 'chromadb', 'redis'"
    )
