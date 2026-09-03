"""Vector database client implementations.

Re-exports the concrete clients along with the abstract base class and
the configuration-driven factory.
"""
from megamem.db_clients.base import VectorDBClient
from megamem.db_clients.chromadb_client import ChromaDBClient
from megamem.db_clients.redis_client import RedisVectorDBClient
from megamem.db_clients.factory import create_vector_db_client

__all__ = [
    "VectorDBClient",
    "ChromaDBClient",
    "RedisVectorDBClient",
    "create_vector_db_client",
]
