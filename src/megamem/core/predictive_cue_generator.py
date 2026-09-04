"""Predictive Cue Anchor (PCA) Generator"""

from typing import Dict, List, Optional

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from megamem.core.general_api import GeneralBadRequestError
from megamem.utils.llm import ChatCompletionModel

logger = logging.getLogger(__name__)


PROMPT_PREDICTIVE_CUE = """\
You are a memory-indexing assistant that generates **predictive retrieval cues**.

# CONTEXT
A personal-memory system stores facts about a user.  Each fact already has a
short *Primary Index* and optional *Intrinsic Cues* that describe what the
memory IS about.  Your job is to produce **Predictive Cues** — short phrases
that describe real-world CONTEXTS in which this memory would be relevant,
even if those contexts seem unrelated on the surface.

# TASK
1. **Classify** the memory into one category:
   - health_constraint  (allergies, medical conditions, medications, mobility)
   - strong_preference  (dietary laws, fears, strong likes/dislikes)
   - identity_fact      (profession, family, location, language)
   - episodic_event     (past trips, purchases, meetings)
   - transient_state    (current mood, temporary situation)

2. **Generate implication chains** of TWO kinds.

   **Kind A — Activity / Decision cues** (what tasks is this relevant for?):
   a. What does this fact IMPLY about real-world situations?
   b. What ACTIVITIES or DECISIONS would be affected?
   c. What could go WRONG if someone making a recommendation or plan did NOT
      know this fact?
   d. What QUERIES might a user ask where this fact should influence the answer?

   **Kind B — Behavioral Consequence cues** (what would the person say, feel,
   or do in the future BECAUSE of this memory?):
   a. How might this fact CHANGE the person's behavior in daily life?
   b. What might the person SAY to others that reveals this memory shaped them?
   c. What EMOTIONS or REACTIONS might they have in situations that echo this
      memory — e.g., anxiety, nostalgia, protectiveness, reluctance?
   d. If someone observed the person acting differently, what STATEMENT would
      the person make to explain why?

3. **Extract a cue** from each chain — a concise 2-5 word phrase describing
   the retrieval context (NOT the memory content).

# EXAMPLES (Kind B)
Memory: "After hurting my back moving apartments, I insist on hiring movers"
  Kind A cues: moving logistics, renovation planning, budget tradeoff
  Kind B cues: warning others about lifting, cautious about physical strain,
               advising against solo hauling

Memory: "I removed all social apps because I value my attention"
  Kind A cues: productivity tools, device setup, deep work planning
  Kind B cues: feeling out of the loop socially, missing shared references,
               explaining why unreachable online

Memory: "Since my dad had a heart attack at 52, I do 30 min cardio daily"
  Kind A cues: morning availability, fitness gear, travel itinerary planning
  Kind B cues: health anxiety from family history, fear of skipping exercise,
               doctor suggesting less exercise

# GUIDELINES
- health_constraint / strong_preference → generate 5-8 chains (mix of A and B)
- identity_fact → 3-5 chains (mix of A and B)
- episodic_event → 2-4 chains (at least 1 of Kind B)
- transient_state → 0 chains (return empty list)
- Ensure at LEAST 2 chains are Kind B (behavioral consequence).
- Do NOT generate cues that are semantically similar to the memory itself or
  to the provided intrinsic cues.
- Each cue must target a DIFFERENT activity / decision domain.

# INPUT
Primary Index: {index}
Memory Value: {value}
Intrinsic Cues: {intrinsic_cues}

Produce your answer as structured JSON (category + list of chains).
"""


class ImplicationChain(BaseModel):
    reasoning: str = Field(description="Brief reasoning chain connecting the memory to a retrieval context")
    cue: str = Field(description="2-5 word phrase describing the anticipated retrieval context")


class PredictiveCueOutput(BaseModel):
    category: str = Field(description="Memory category: health_constraint, strong_preference, identity_fact, episodic_event, or transient_state")
    chains: List[ImplicationChain] = Field(default_factory=list, description="Implication chains with extracted cues")


_sentence_model = None


def _get_sentence_model():
    """Lazily instantiate a small sentence-transformer used for cue filtering."""
    global _sentence_model
    if _sentence_model is not None:
        return _sentence_model
    try:
        import os
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("LOCOMO_SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2")
        local_only = os.getenv("LOCOMO_SENTENCE_TRANSFORMERS_LOCAL_ONLY", "").lower() in {"1", "true", "yes"}
        _sentence_model = SentenceTransformer(model_name, local_files_only=local_only)
        return _sentence_model
    except Exception as e:
        logger.warning("Could not load SentenceTransformer for PCA filter: %s", e)
        return None


_OVER_GENERIC_PHRASES = frozenset({
    "daily life", "general planning", "everyday activities",
    "personal life", "general advice", "life decisions",
    "future planning", "general conversation", "daily routine",
    "common activities",
})


def _cosine_sim_batch(model, texts_a: List[str], texts_b: List[str]):
    """For each text in *texts_a*, return its max cosine similarity against any text in *texts_b*."""
    import numpy as np
    from sentence_transformers.util import pytorch_cos_sim

    if not texts_a or not texts_b:
        return [0.0] * len(texts_a)

    emb_a = model.encode(texts_a, convert_to_tensor=True)
    emb_b = model.encode(texts_b, convert_to_tensor=True)
    sim_matrix = pytorch_cos_sim(emb_a, emb_b)  # shape: (len_a, len_b)
    return sim_matrix.max(dim=1).values.cpu().numpy().tolist()


class PredictiveCueGenerator:
    """Produce extrinsic (predictive) cue indices for a memory entry."""

    REDUNDANCY_THRESHOLD = 0.75

    def __init__(self, cfg: DictConfig, model_client: ChatCompletionModel):
        self.cfg = cfg
        self._model_client = model_client

    def generate(
        self,
        index: str,
        value: str,
        intrinsic_cues: List[str],
    ) -> List[str]:
        """Produce predictive cues for a single memory.

        Returns a deduplicated, quality-filtered list of extrinsic cue
        phrases (possibly empty for transient-state memories).
        """
        try:
            output = self._generate_chains(index, value, intrinsic_cues)
        except Exception:
            logger.warning("PCA generation failed for '%s', returning empty cues", index, exc_info=True)
            return []

        raw_cues = [chain.cue.strip() for chain in output.chains if chain.cue.strip()]
        if not raw_cues:
            return []

        return self._filter_cues(raw_cues, intrinsic_cues, index)

    def _generate_chains(
        self, index: str, value: str, intrinsic_cues: List[str]
    ) -> PredictiveCueOutput:
        intrinsic_str = ", ".join(intrinsic_cues) if intrinsic_cues else "(none)"
        prompt_args = {
            "index": index,
            "value": value,
            "intrinsic_cues": intrinsic_str,
        }
        # ChatCompletionModel already injects seed from cfg.llm.seed;
        # we only pass temperature here to avoid duplicate-keyword errors.
        kwargs: Dict = {"temperature": 0.7}

        try:
            result = self._model_client.invoke(
                input=PROMPT_PREDICTIVE_CUE,
                prompt_args=prompt_args,
                response_format=PredictiveCueOutput,
                source="PredictiveCueGenerator",
                **kwargs,
            )
        except GeneralBadRequestError as e:
            err_text = str(e).lower()
            if "temperature" in err_text or "unsupported_value" in err_text:
                logger.info("Retrying PCA without temperature for '%s'", index)
                kwargs.pop("temperature", None)
                result = self._model_client.invoke(
                    input=PROMPT_PREDICTIVE_CUE,
                    prompt_args=prompt_args,
                    response_format=PredictiveCueOutput,
                    source="PredictiveCueGenerator",
                    **kwargs,
                )
            else:
                raise

        return result

    def _filter_cues(
        self,
        raw_cues: List[str],
        intrinsic_cues: List[str],
        primary_index: str,
    ) -> List[str]:
        """Drop redundant, over-generic, or duplicate cues."""
        # 1. Case-insensitive deduplication.
        seen_lower: set = set()
        unique: List[str] = []
        for cue in raw_cues:
            low = cue.lower()
            if low in seen_lower:
                continue
            seen_lower.add(low)
            unique.append(cue)

        # 2. Strip out over-generic phrases.
        filtered = [
            c for c in unique
            if c.lower() not in _OVER_GENERIC_PHRASES and len(c.split()) >= 2
        ]

        # 3. Embedding-based redundancy check vs intrinsic cues + primary index.
        model = _get_sentence_model()
        if model is not None and (intrinsic_cues or primary_index):
            reference_texts = list(intrinsic_cues) + [primary_index]
            max_sims = _cosine_sim_batch(model, filtered, reference_texts)
            filtered = [
                cue for cue, sim in zip(filtered, max_sims)
                if sim < self.REDUNDANCY_THRESHOLD
            ]

        return filtered
