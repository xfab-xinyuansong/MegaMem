# API Guide

MegaMem separates its lightweight contracts and remote client from optional local
retrieval, model, and document-processing dependencies. This guide describes the public
interfaces and the installation required for each one.

## Installation Profiles

| Profile | Command | Provides |
| --- | --- | --- |
| Core | `pip install -e .` | Data contracts, token accounting, CLI, memory client, general API client |
| Local memory | `pip install -e ".[retrieval,llm,documents]"` | Vector retrieval, environment loading, builders, file ingestion |
| Local models | `pip install -e ".[local-models]"` | Local causal-model execution |
| Evaluation | `pip install -e ".[evaluation]"` | Evaluation metrics |
| Hugging Face data | `pip install -e ".[huggingface]"` | Streaming loaders for the EnterpriseRAG extension |
| Development | `pip install -e ".[dev]"` | Tests and package builds |

Optional backends are imported only when their interfaces are used. A core installation
can therefore import `megamem`, run the CLI, and construct a remote client without
installing the local retrieval stack.

## Hugging Face Dataset

The optional dataset loaders expose the document and question configurations without
requiring the complete retrieval stack:

```python
from megamem import load_enterprise_rag

benchmark = load_enterprise_rag()  # streams both configurations by default
document = next(iter(benchmark["documents"]))
question = next(iter(benchmark["questions"]))
```

Documents contain `doc_id`, `source_type`, `title`, and `content`. Questions contain
the query and reference answer together with expected document identifiers and answer
facts. Pass `streaming=False` to materialize Arrow datasets locally, or `revision` to
pin a specific Hugging Face snapshot.

## Core Contracts

The package root exposes the interfaces supported by every installation:

```python
from megamem import (
    DualNode,
    DualNodeError,
    GeneralAPIClient,
    GeneralAPIError,
    MemoryClient,
    TokenLedger,
    validate_batch,
    validate_one,
)
```

`DualNode` stores compact and detailed representations with stable source-evidence
identifiers. `validate_one` and `validate_batch` enforce representation and provenance
invariants. `TokenLedger` records per-stage token and latency measurements without
requiring model credentials. `GeneralAPIClient` provides the package-owned JSON model
gateway; `MemoryClient` connects applications to an MegaMem memory service.

## Remote Client

Remote mode needs only the core installation and an MegaMem service endpoint:

```python
import os

from megamem import MemoryClient

client = MemoryClient(
    api_key=os.environ["MEGAMEM_API_KEY"],
    server_url=os.environ.get("MEGAMEM_SERVER_URL", "http://localhost:8000"),
)

client.add("Evidence to retain", metadata={"source_id": "document:42"})
results = client.query("What evidence was retained?", top_k=5)
planned = client.planner_query("Find the governing source", top_k=5)
```

Remote mode supports `add`, `query`, and `planner_query`. File processing, custom
builders, callbacks, record administration, and named retrieval strategies are local
operations. Network failures and invalid JSON responses raise `RemoteMemoryError` with
the request method and URL while preserving the original exception as the cause.

## Local Client

Local mode requires the local-memory installation profile and an `OmegaConf`
`DictConfig` containing the memory, retrieval, embedding, and model settings consumed by
the selected pipeline:

```python
from omegaconf import OmegaConf

from megamem import MemoryClient

cfg = OmegaConf.load("path/to/runtime.yaml")
client = MemoryClient(cfg=cfg, user_id="workspace-user")

client.add("A source-grounded memory record")
client.add_file("documents/policy.pdf", metadata={"source_id": "policy:2026"})
results = client.query("policy requirements", top_k=10)
```

The user identifier scopes the local collection. `advanced_query` accepts `semantic`,
`prompt`, `plan`, `reformulate`, or `hybrid` as its `query_type`. `list_memories`, `get`,
`delete`, `count`, `clear`, and `delete_all` operate on the local store only.

## Optional Interfaces

Specialized components remain available through explicit imports:

```python
from megamem import DualIndex
from megamem.methods import build_l0_dualnodes
```

`DualIndex` requires the `retrieval` extra. Hierarchy construction may additionally
require the model configuration used by the selected builder. Explicit optional imports
fail at the point of use when their dependency group is absent; core imports remain
available.

## General Model Gateway

Hosted model traffic uses a single public provider alias, `general`, for chat and
embedding endpoints. Keep credentials in environment variables and model aliases in
`configs/models.yaml`:

```bash
cp .env.example .env
cp configs/models.example.yaml configs/models.yaml
megamem doctor --strict
megamem config
```

The resolver fails before network access when an alias is missing, still contains a
placeholder model, or declares any provider other than `general`. The CLI reports
endpoint and alias status without printing API keys. Do not commit `.env`, private
corpora, generated indexes, or raw model outputs.

Hosted chat calls use `POST /chat/completions`; hosted embeddings use
`POST /embeddings`. Both receive bearer authentication. Local embeddings remain the
default and do not require API credentials. Provider routing, tenancy, and account-level
deployment details remain behind the configured general endpoint. The client has no
vendor SDK dependency.
