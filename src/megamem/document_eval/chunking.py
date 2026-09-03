"""
Document → Section → Chunk segmentation utilities.

For EnterpriseRAG documents (slack threads, gmail, confluence, github issues,
etc.) we use a simple but deterministic structural splitter:
- Split on markdown-ish headings (##, ###) for confluence/wiki content
- Split on blank-line paragraphs as fallback
- Sub-chunk paragraphs that exceed chunk_target_tokens
- Each chunk records section_path / section_id / chunk_id deterministically

For Slack/gmail-style content with no headings, the entire document is one
section; chunks are token-bounded paragraphs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_ENC.encode(text))


@dataclass
class ChunkSpec:
    chunk_id: str
    section_id: str
    section_path: str
    position: int
    raw_text: str
    token_count: int


@dataclass
class SectionSpec:
    section_id: str
    section_path: str
    title: str
    level: int
    chunks: List[ChunkSpec] = field(default_factory=list)
    parent_section_id: str = ""
    prev_section_id: str = ""
    next_section_id: str = ""


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def split_into_sections(text: str) -> List[tuple]:
    """Return list of (level, title, body_text) tuples.

    If no headings found, returns a single ('', 0, text) tuple.
    """
    if not text:
        return [("", 0, "")]

    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("", 0, text.strip())]

    sections: List[tuple] = []
    # Prefix (text before first heading) becomes an "intro" section
    if matches[0].start() > 0:
        prefix = text[: matches[0].start()].strip()
        if prefix:
            sections.append(("Intro", 0, prefix))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((title, level, body))

    return sections


def chunk_section(
    body: str,
    target_tokens: int,
    overlap_tokens: int = 0,
) -> List[str]:
    """Greedy paragraph-aware chunker.

    Splits body into paragraphs (blank-line separated); packs paragraphs into
    chunks until target_tokens is exceeded. Paragraphs larger than target are
    further split by sentence then by token if needed.
    """
    if not body.strip():
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for p in paragraphs:
        p_tokens = count_tokens(p)
        if p_tokens > target_tokens:
            # Flush current
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            # Split big paragraph by sentences
            sentences = re.split(r"(?<=[.!?])\s+", p)
            sub_current: List[str] = []
            sub_tokens = 0
            for s in sentences:
                s_tok = count_tokens(s)
                if sub_tokens + s_tok > target_tokens and sub_current:
                    chunks.append(" ".join(sub_current))
                    sub_current, sub_tokens = [s], s_tok
                else:
                    sub_current.append(s)
                    sub_tokens += s_tok
            if sub_current:
                chunks.append(" ".join(sub_current))
            continue
        if current_tokens + p_tokens > target_tokens and current:
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(p)
        current_tokens += p_tokens

    if current:
        chunks.append("\n\n".join(current))

    # Final safety: hard-truncate any chunk that still exceeds 2x target tokens
    safe: List[str] = []
    for c in chunks:
        toks = _ENC.encode(c)
        if len(toks) > target_tokens * 2:
            for i in range(0, len(toks), target_tokens):
                safe.append(_ENC.decode(toks[i : i + target_tokens]))
        else:
            safe.append(c)
    return [c for c in safe if c.strip()]


def segment_document(
    document_id: str,
    text: str,
    target_tokens: int = 400,
    overlap_tokens: int = 0,
    max_chunks: int = 200,
) -> List[SectionSpec]:
    """Top-level: split a document into sections and chunks.

    Returns ordered list of SectionSpec, each with ordered chunks. chunk_id
    format: ``"{document_id}__sec_{section_idx}__chunk_{chunk_idx}"``.
    """
    raw_sections = split_into_sections(text)
    out: List[SectionSpec] = []
    chunk_global_count = 0

    for sec_idx, (title, level, body) in enumerate(raw_sections):
        section_id = f"{document_id}__sec_{sec_idx}"
        section_path = title or f"Section {sec_idx}"
        section = SectionSpec(
            section_id=section_id,
            section_path=section_path,
            title=title,
            level=level,
        )
        chunk_texts = chunk_section(body, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
        for ci, ctext in enumerate(chunk_texts):
            if chunk_global_count >= max_chunks:
                break
            chunk = ChunkSpec(
                chunk_id=f"{section_id}__chunk_{ci}",
                section_id=section_id,
                section_path=section_path,
                position=ci,
                raw_text=ctext,
                token_count=count_tokens(ctext),
            )
            section.chunks.append(chunk)
            chunk_global_count += 1
        if section.chunks:  # skip empty sections
            out.append(section)
        if chunk_global_count >= max_chunks:
            break

    # Wire prev/next links between sections
    for i, sec in enumerate(out):
        if i > 0:
            sec.prev_section_id = out[i - 1].section_id
        if i + 1 < len(out):
            sec.next_section_id = out[i + 1].section_id

    return out
