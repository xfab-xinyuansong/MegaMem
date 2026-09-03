# MegaMem: A Retrieval Solution for Ultra-Large Context Windows

**A retrieval solution for ultra-large context windows**

<p align="center"><strong>Developed by <a href="https://cielara.ai">Cielara AI</a></strong></p>

<p align="center">
  <a href="https://cielara.ai"><img alt="Cielara AI" src="https://img.shields.io/badge/Cielara-AI-6D28D9.svg"></a>
  <a href="https://arxiv.org/abs/2608.22137"><img alt="arXiv 2608.22137" src="https://img.shields.io/badge/arXiv-2608.22137-B31B1B.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="pyproject.toml"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-blue"></a>
</p>

Modern models and agents increasingly need persistent memory for complete codebases,
long interaction histories, and heterogeneous enterprise records. Yet only a small
fraction of that memory supports any one answer. Loading more content increases cost and
distractor exposure, while compressed records can omit the dates, exceptions, and
conflicts required for a correct response.

MegaMem separates the full **persistent context** from the bounded **evidence context**
used for one answer. Original and transformed queries search detailed evidence and
distilled typed memories in parallel. Every distilled hit resolves to an immutable
source ID before fusion, deduplication, and reranking; only the highest-ranked detailed
evidence within a fixed budget reaches the answer model.

<p align="center">
  <img src="assets/megamem_overview_evidence_context.png" width="100%" alt="MegaMem maps ultra-large external memory to a bounded evidence context">
</p>

<p align="center"><em>The persistent context can grow to approximately 650M tokens while each query loads only a small, source-resolved evidence context for answering.</em></p>

## Multi-Route Recall and Post-Answer Attribution

<p align="center">
  <img src="assets/megamem_recall_attribution_answering.png" width="100%" alt="MegaMem recall, evidence selection, answering, and post-answer attribution">
</p>

<p align="center"><em>Both memory views are searched with original and transformed queries; reciprocal-rank fusion, deduplication, and cross-encoder reranking select detailed evidence, and attribution identifies supporting sources only after the answer is fixed.</em></p>

## Core Guarantees

- **Dual-view memory.** Each `DualNode` pairs a compact retrieval representation with
  detailed source evidence and stable provenance identifiers.
- **Source-resolved generation.** Distilled hits are mapped back to authoritative text
  before reranking, packing, and answer generation.
- **Bounded evidence context.** Candidate depth and answer-context size are independent,
  allowing persistent memory to grow without expanding every prompt.
- **Post-answer attribution.** Reported sources are reduced after generation without
  rewriting the fixed answer.
- **General model gateway.** Every hosted model call resolves through the public
  `general` alias; endpoint, model, and credentials remain deployment configuration.

## Architecture

| Stage | Input | Output | Contract |
| --- | --- | --- | --- |
| Build | Authorized source spans | Distilled and detailed views | Preserve source identifiers |
| Recall | Original and transformed queries | Multi-route candidates | Search both views |
| Resolve | Distilled candidates | Authoritative source text | Never answer from a summary alone |
| Select | Source-text candidates | Ranked evidence set | Fuse, deduplicate, and rerank |
| Pack | Ranked evidence | Bounded evidence context | Enforce an explicit token budget |
| Answer | Packed detailed evidence | Fixed answer | Generate only from source-resolved evidence |
| Attribute | Fixed answer and loaded evidence | Supporting source IDs | Select sources without rewriting the answer |

The key distinction is between **persistent context** and **evidence context**. The
former can contain approximately 650M tokens and grow toward one billion; the latter
remains a small, inspectable evidence budget for one query.

## Installation

MegaMem requires Python 3.11 or newer.

```bash
git clone https://github.com/xfab-xinyuansong/MegaMem.git
cd MegaMem

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

The core installation provides the dual-view data contracts, token ledger, CLI, memory
service client, and general model-gateway client without loading a vector database or
local model runtime. Add only the capability groups needed for a deployment:

```bash
pip install -e ".[retrieval,llm,documents]"  # complete local memory pipeline
pip install -e ".[local-models]"             # optional local model execution
pip install -e ".[evaluation]"               # evaluation metrics
pip install -e ".[dev]"                      # tests and package build tools
```

## Quick Start

The zero-credential example validates the representation and provenance contracts:

```bash
python examples/minimal_contract.py
```

The same primitives are available from the package root:

```python
from megamem import DualNode, TokenLedger, validate_batch

node = DualNode(
    node_id="policy:001",
    level="L0",
    distilled_text="Travel reimbursement policy and approval rules.",
    detailed_text="Employees must submit receipts within 30 days of travel.",
    distilled_tokens=7,
    detailed_tokens=10,
    source_evidence_ids=["policy:001#section-4"],
)

report = validate_batch([node])
assert report["overall_pass"]

ledger = TokenLedger(run_id="demo", method="dual-view")
ledger.record("retrieval", "local", input_tokens=7, output_tokens=0, wall_seconds=0.01)
print(ledger.grand_total())
```

## Memory Service API

MegaMem keeps its two network boundaries separate:

| Boundary | Purpose | Configuration |
| --- | --- | --- |
| Memory service | Add and query persistent memory from an application | `MEGAMEM_SERVER_URL`, `MEGAMEM_API_KEY` |
| General model gateway | Build, rerank, judge, and answer with hosted models | Model aliases plus general API environment variables |

`MemoryClient` provides one entry point for deployed and in-process memory operation. A
core installation can connect to a MegaMem service without importing local model,
document, or vector-store dependencies:

```python
import os

from megamem import MemoryClient

memory = MemoryClient(
    api_key=os.environ["MEGAMEM_API_KEY"],
    server_url=os.environ["MEGAMEM_SERVER_URL"],
)
memory.add(
    "Quarterly access reviews are due by the final business day.",
    metadata={"source_id": "policy:access-review"},
)
matches = memory.query("When is the access review due?", top_k=5)
```

Local operation accepts a `DictConfig` plus a user-scoped identifier and exposes file,
chat, email, planner, and advanced retrieval workflows. See the [API guide](docs/API.md)
for supported operations, optional dependencies, and error behavior.

## General Model Gateway

Copy the public templates and provide the endpoints used by your run:

```bash
cp .env.example .env
cp configs/models.example.yaml configs/models.yaml
```

```dotenv
LLM_API_BASE=https://your-endpoint.example/v1
LLM_API_KEY=your-secret
LLM_CHAT_MODEL=your-chat-model
LLM_JUDGE_MODEL=your-judge-model

# Optional hosted embeddings; local embeddings are the default.
MEGAMEM_LOCAL_EMBEDDING=0
EMBEDDING_API_BASE=https://your-endpoint.example/v1
EMBEDDING_API_KEY=your-secret
EMBEDDING_MODEL=your-embedding-model
```

Model names live in `configs/models.yaml`; API keys stay in `.env`. MegaMem exposes one
hosted provider alias, `general`. The resolver rejects missing aliases, placeholder
models, and any other provider value before a request is sent. Set
`MEGAMEM_MODELS_CONFIG` when the alias file is outside the repository root.

The package also exposes the same gateway contract directly, without a vendor SDK:

```python
import os

from megamem import GeneralAPIClient

gateway = GeneralAPIClient(
    base_url=os.environ["LLM_API_BASE"],
    api_key=os.environ["LLM_API_KEY"],
)
response = gateway.chat.completions.create(
    model=os.environ["LLM_CHAT_MODEL"],
    messages=[{"role": "user", "content": "Summarize the selected evidence."}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

All hosted model traffic follows one general contract:

| Operation | Method and path | Required request fields |
| --- | --- | --- |
| Chat | `POST /chat/completions` | `model`, `messages`, token limit |
| Embeddings | `POST /embeddings` | `model`, `input` |

The gateway receives bearer authentication. The client uses plain JSON over HTTP and
contains no cloud vendor, account resource, provider-specific deployment field, or
vendor SDK. Routing remains behind the configured endpoint. Local sentence-transformer
embeddings require no API configuration.

Inspect the active, non-secret configuration and optional dependency groups:

```bash
megamem doctor
megamem doctor --json
megamem config
```

Use `megamem doctor --strict` in deployment checks when model aliases must be fully
configured. The command never prints API keys.

## Package Verification

Credential-free checks cover package imports, CLI behavior, dual-view invariants,
provenance, configuration isolation, and token accounting:

```bash
make check
```

| Package check | Entry point |
| --- | --- |
| Package and CLI | `pytest` |
| Representation and provenance | `python -m megamem.methods.dual_node` |
| Token accounting | `python -m megamem.methods.token_ledger` |

## Repository Layout

```text
MegaMem/
├── src/megamem/
│   ├── methods/          # dual nodes, indexing, hierarchy construction, token ledger
│   ├── document_eval/    # ingestion, retrieval, answering, and metrics
│   ├── retriever/        # semantic, hybrid, planning, and reformulation strategies
│   ├── builder/          # document, chat, and email memory builders
│   ├── processors/       # text, PDF, Word, PowerPoint, Excel, and Markdown readers
│   ├── core/             # memory entries, stores, filters, planners, and source cues
│   └── db_clients/       # vector-store and cache adapters
├── configs/              # public model-routing templates
├── examples/             # runnable, credential-free examples
├── tests/                # package and method checks
└── docs/                 # package API documentation
```

## Package Scope

This repository contains the installable implementation and public package contracts. It does not ship
private or licensed corpora, generated vector indexes, raw model outputs, real
credentials, provider account configuration, manuscript sources, plotting utilities, or
benchmark result artifacts.

- [API guide](docs/API.md)
- [Contribution guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## Citation

[Paper](https://arxiv.org/abs/2608.22137) · [PDF](https://arxiv.org/pdf/2608.22137)

```bibtex
@misc{song2026megamem,
  title         = {{MegaMem}: A Retrieval Solution for Ultra-Large Context Windows},
  author        = {Xinyuan Song and Bowen Zhu and Hasibul Haque and Liang Zhao},
  year          = {2026},
  eprint        = {2608.22137},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2608.22137},
  url           = {https://arxiv.org/abs/2608.22137}
}
```

## License

MegaMem is released under the [MIT License](LICENSE).
