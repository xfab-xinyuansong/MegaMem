"""Configuration-driven factory for vector database clients."""
from omegaconf import DictConfig

from megamem.db_clients.base import VectorDBClient
from megamem.db_clients.chromadb_client import ChromaDBClient
from megamem.db_clients.redis_client import RedisVectorDBClient


def create_vector_db_client(cfg: DictConfig) -> VectorDBClient:
    """Build a concrete ``VectorDBClient`` from configuration."""
    db_type = cfg.memory.get("db_type", "chromadb").lower()

    if db_type in ("chromadb", "chroma"):
        return ChromaDBClient(cfg)
    if db_type == "redis":
        return RedisVectorDBClient(cfg)

    raise ValueError(
        f"Unsupported database type: {db_type}. "
        f"Supported types: 'chromadb', 'redis'"
    )
