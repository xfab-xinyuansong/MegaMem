import json
from types import SimpleNamespace

import pytest

from megamem.experiments.backend import PersistentIndex
from megamem.experiments.build import build_identity, chunk_document, validate_abstraction, validate_atomic
from megamem.experiments.runtime import ABSTENTION, PaperRuntime, evidence_card, pack_evidence, resolve_and_fuse


class CharacterCounter:
    def count(self, text):
        return len(text)


def configuration(tmp_path, **retrieval):
    return {"index": {"path": str(tmp_path)}, "retrieval": {"rerank": False, **retrieval}}


def raw_chunk(identifier, doc_id, content):
    return {
        "chunk_id": identifier, "doc_id": doc_id, "content": content,
        "section_path": "Section", "representation": "raw", "source_chunk_ids": [identifier],
    }


def test_runtime_construction_is_lazy_and_build_does_not_require_reranker(tmp_path):
    runtime = PaperRuntime({"index": {"path": str(tmp_path / "does-not-exist")}})
    assert runtime._store is None
    assert runtime._reranker is None
    assert runtime._clients == {}
    assert runtime.tokens._encoding is None
    assert not (tmp_path / "does-not-exist").exists()


def test_resolve_precedes_deduplication_and_reciprocal_rank_fusion():
    routes = [
        {"route_id": "original:distilled", "query": "q", "view": "distilled_memory", "weight": 1.0,
         "hits": [
             {"id": "m1", "source_chunk_ids": ["a"]},
             {"id": "m2", "source_chunk_ids": ["a"]},
             {"id": "m3", "source_chunk_ids": ["b"]},
         ]},
        {"route_id": "original:raw", "query": "q", "view": "raw_chunks", "weight": 1.0,
         "hits": [{"id": "b", "source_chunk_ids": ["b"]}]},
    ]
    ranking, scores, trace = resolve_and_fuse(routes, 60)
    assert ranking == ["b", "a"]
    assert scores["a"]["rrf_score"] == pytest.approx(1 / 61)
    assert scores["b"]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert trace[0]["resolved_chunk_ids"] == ["a", "b"]


def test_multi_source_abstraction_resolves_every_child_before_fusion():
    routes = [{
        "route_id": "canonical", "query": "q", "view": "distilled_memory", "weight": 0.5,
        "hits": [{"id": "group", "source_chunk_ids": ["b", "a"]}],
    }]
    ranking, scores, _ = resolve_and_fuse(routes, 60)
    assert ranking == ["b", "a"]
    assert scores["a"]["rrf_score"] == pytest.approx(0.5 / 62)


def test_packing_counts_labels_and_separators_and_respects_document_limit():
    chunks = [raw_chunk("a", "doc-a", "one"), raw_chunk("b", "doc-b", "two")]
    tokens = CharacterCounter()
    budget = tokens.count(evidence_card(chunks[0]))
    selected, text, skipped = pack_evidence(chunks, tokens, budget, 5)
    assert [chunk["chunk_id"] for chunk in selected] == ["a"]
    assert tokens.count(text) == budget
    assert skipped == [{"chunk_id": "b", "reason": "evidence_token_budget"}]
    selected, _, skipped = pack_evidence(chunks, tokens, 10000, 1)
    assert [chunk["doc_id"] for chunk in selected] == ["doc-a"]
    assert skipped[0]["reason"] == "document_budget"


def test_section_chunks_are_exact_source_substrings_and_bounded():
    source = "# One\n" + "alpha beta " * 15 + "\n## Two\n" + "gamma " * 20
    document = {"doc_id": "d", "content": source}
    chunks = chunk_document(document, CharacterCounter(), 70)
    assert any(chunk["section_path"] == "One / Two" for chunk in chunks)
    assert all(chunk["token_count"] <= 70 for chunk in chunks)
    assert all(chunk["content"] == source[chunk["start_char"]:chunk["end_char"]] for chunk in chunks)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)


def test_typed_memory_limit_and_abstraction_source_identity_are_strict():
    valid = {"type": "fact", "key": "term", "value": "statement"}
    with pytest.raises(ValueError, match="three"):
        validate_atomic({"memories": [valid] * 4})
    with pytest.raises(ValueError, match="identifiers"):
        validate_abstraction({"search_key": "term", "summary": "statement", "source_chunk_ids": ["wrong"]}, ["a"])


def test_gold_answer_is_never_forwarded_and_attribution_cannot_change_prediction(tmp_path):
    runtime = PaperRuntime(configuration(tmp_path, evidence_source="gold", attribution=True))
    runtime.tokens = CharacterCounter()
    prompts = []

    def chat(stage, system, user, limit, json_output=False):
        prompts.append((stage, user))
        return "Evidence-supported answer."

    def attribute(stage, system, user, validator, fallback, limit):
        prompts.append((stage, user))
        return ["doc-a", "unloaded-doc"]

    runtime._chat = chat
    runtime._json_call = attribute
    result = runtime.answer({
        "question_id": "q", "question": "What does the evidence say?", "gold_answer": "SECRET GOLD ANSWER",
        "gold_chunks": [raw_chunk("a", "doc-a", "Only this original evidence may be used.")],
    })
    assert result["prediction"] == "Evidence-supported answer."
    assert result["reported_doc_ids"] == ["doc-a"]
    assert result["trace"]["attribution_rejected_doc_ids"] == ["unloaded-doc"]
    assert all("SECRET GOLD ANSWER" not in prompt for _, prompt in prompts)
    assert [stage for stage, _ in prompts] == ["answer", "attribution"]
    assert runtime._store is None


def test_explicit_empty_gold_evidence_abstains_without_api(tmp_path):
    runtime = PaperRuntime(configuration(tmp_path, evidence_source="gold"))
    runtime.tokens = CharacterCounter()
    result = runtime.answer({"question_id": "q", "question": "Unanswerable?", "gold_chunks": []})
    assert result["status"] == "ok"
    assert result["prediction"] == ABSTENTION
    assert result["loaded_doc_ids"] == []
    assert runtime._clients == {}


def test_incomplete_index_is_rejected_before_any_api(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "incomplete"}), encoding="utf-8")
    runtime = PaperRuntime(configuration(tmp_path))
    result = runtime.answer({"question_id": "q", "question": "Question"})
    assert result["status"] == "error"
    assert "completed" in result["errors"][0]["message"]
    assert runtime._clients == {}


def test_inference_collection_uses_get_and_preserves_scope(tmp_path):
    calls = []

    class Collection:
        metadata = {"hnsw:space": "cosine"}

        def count(self):
            return 10

        def query(self, **kwargs):
            calls.append(kwargs)
            return {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}

    def get_collection(**kwargs):
        calls.append(kwargs)
        return Collection()

    index = PersistentIndex({"index": {"path": str(tmp_path)}})
    index._chroma = SimpleNamespace(get_collection=get_collection)
    assert index.query("raw_chunks", [0.1], 5, ["local-doc"]) == []
    assert calls[0] == {"name": "raw_chunks", "embedding_function": None}
    assert calls[1]["where"] == {"doc_id": {"$in": ["local-doc"]}}


def test_bm25_does_not_leak_outside_question_corpus(tmp_path):
    index = PersistentIndex({"index": {"path": str(tmp_path)}}, build_mode=True)
    try:
        for position, doc_id in enumerate(["local", "other"]):
            index.stage_document(
                {"doc_id": doc_id, "fingerprint": doc_id, "position": position, "token_count": 2},
                [raw_chunk(doc_id + "-chunk", doc_id, "alpha beta")], [],
            )
            index.finish_document(doc_id)
        hits = index.bm25("alpha", 20, 1.5, 0.75, ["local"])
        assert [hit["doc_id"] for hit in hits] == ["local"]
        assert index.bm25("alpha", 20, 1.5, 0.75, []) == []
    finally:
        index.close()


def test_build_identity_is_extractor_isolated_but_retrieval_independent(tmp_path):
    first = PaperRuntime(configuration(tmp_path))
    second = PaperRuntime(configuration(tmp_path, query_expansion=False, attribution=False))
    assert build_identity(first.config) == build_identity(second.config)
    second.config["index"]["extractor"] = "full"
    assert build_identity(first.config) != build_identity(second.config)
