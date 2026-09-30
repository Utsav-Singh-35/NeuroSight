"""Cited clinical knowledge base loader.

Replaces the hardcoded ``TUMOR_INFO`` dict with markdown files under
``knowledge/`` in which **every clinical statement carries a source ID**. That is
the difference between a report that sounds authoritative and one that is
evidence-grounded: each sentence can be traced back to a named publication.

Design choices
--------------
*Markdown with a small frontmatter block*, rather than JSON, because the content
is prose meant to be read and reviewed by a human before deployment. Reviewing a
JSON blob full of escaped newlines is unpleasant enough that it would not happen.

*A hand-written frontmatter parser*, rather than PyYAML, because the schema is
five scalar keys and one list. Adding a dependency for that would not be worth
it, and PyYAML is not currently installed.

*Sections double as RAG chunks.* ``## Overview``, ``## Warning signs`` and so on
are already semantically coherent units of roughly the right size, so retrieval
does not need an arbitrary sliding window. :meth:`KnowledgeBase.chunks` emits
them directly with their metadata attached.

Safety behaviour
----------------
The previous implementation did ``TUMOR_INFO.get(prediction, TUMOR_INFO["No Tumor"])``,
which silently rendered a reassuring "no tumour detected" narrative for *any*
unrecognised label. Renaming a class, or adding a module, would have produced
confidently wrong reassurance. :meth:`KnowledgeBase.get` returns ``None`` for an
unknown label and callers must handle that explicitly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Inline citation markers such as [nci-cns-pdq-hp]. Full-width CJK brackets are
# normalised first because LLM-authored text sometimes uses them, and an
# unmatched marker would silently drop a citation.
_CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9\-\.]*)\]")
_BRACKET_TRANSLATION = str.maketrans({"【": "[", "】": "]", "〔": "[", "〕": "]"})


def normalise_citation_brackets(text: str) -> str:
    """Convert non-ASCII citation brackets to ASCII square brackets."""
    return text.translate(_BRACKET_TRANSLATION)

# Section heading -> stable machine key.
_SECTION_KEYS = {
    "overview": "overview",
    "interpretation": "interpretation",
    "interpretation - read this before reassuring anyone": "interpretation",
    "ai limitations for this class": "ai_limitations",
    "relevant investigations": "investigations",
    "clinical considerations": "clinical_considerations",
    "follow-up considerations": "follow_up",
    "warning signs": "warning_signs",
    "treatment information": "treatment_information",
}


def _slugify_heading(heading: str) -> str:
    """Normalise a markdown heading for lookup in ``_SECTION_KEYS``."""
    cleaned = heading.strip().lower()
    cleaned = cleaned.replace("\u2014", "-").replace("\u2013", "-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -")


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a leading ``---`` frontmatter block from the body.

    Supports ``key: value`` scalars and ``key: [a, b, c]`` lists. Booleans and
    integers are coerced; everything else stays a string.

    Args:
        text: Full file contents.

    Returns:
        ``(metadata, body)``. Metadata is empty if no frontmatter is present.
    """
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    meta: dict = {}
    for line in parts[1].strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        raw = raw.strip()

        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1].strip()
            meta[key] = [v.strip() for v in inner.split(",") if v.strip()]
        elif raw.lower() in {"true", "false"}:
            meta[key] = raw.lower() == "true"
        elif raw.isdigit():
            meta[key] = int(raw)
        else:
            meta[key] = raw

    return meta, parts[2].lstrip("\n")


def _parse_sections(body: str) -> dict[str, dict]:
    """Split a markdown body into ``## `` sections.

    Returns:
        ``key -> {"heading": str, "text": str, "citations": [ids]}``
    """
    body = normalise_citation_brackets(body)
    sections: dict[str, dict] = {}
    current_heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_heading is None:
            return
        text = "\n".join(buffer).strip()
        if not text:
            return
        slug = _slugify_heading(current_heading)
        key = _SECTION_KEYS.get(slug, re.sub(r"[^a-z0-9]+", "_", slug).strip("_"))
        sections[key] = {
            "heading": current_heading.strip(),
            "text": text,
            "citations": sorted(set(_CITATION_RE.findall(text))),
        }

    for line in body.splitlines():
        if line.startswith("## "):
            flush()
            current_heading = line[3:]
            buffer = []
        else:
            buffer.append(line)

    flush()
    return sections


@dataclass
class KnowledgeEntry:
    """One class's clinical knowledge, with citations."""

    class_label: str
    module_id: str
    risk_tier: str
    requires_clinical_review: bool
    sources: list[str]
    sections: dict[str, dict] = field(default_factory=dict)
    path: str = ""

    def section_text(self, key: str) -> str | None:
        """Raw markdown text of a section, or None."""
        section = self.sections.get(key)
        return section["text"] if section else None

    def plain(self, key: str) -> str | None:
        """Section text with citation markers and markdown emphasis stripped.

        Used for report fields that are rendered as prose rather than markdown.
        """
        text = self.section_text(key)
        if text is None:
            return None
        stripped = _CITATION_RE.sub("", text)
        stripped = stripped.replace("**", "").replace("`", "")
        stripped = re.sub(r"[ \t]+", " ", stripped)
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        return stripped.strip()

    def all_citations(self) -> list[str]:
        """Every source ID cited anywhere in this entry."""
        found: set[str] = set(self.sources)
        for section in self.sections.values():
            found.update(section["citations"])
        return sorted(found)


class KnowledgeBase:
    """Loads and indexes the cited knowledge base."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.sources: dict = {}
        self.source_meta: dict = {}
        self.entries: dict[tuple[str, str], KnowledgeEntry] = {}
        self.available = False
        self._load()

    def _load(self) -> None:
        if not self.root.is_dir():
            logger.warning("Knowledge base directory not found: %s", self.root)
            return

        sources_path = self.root / "sources.json"
        if sources_path.exists():
            import json

            with open(sources_path, "r", encoding="utf-8") as handle:
                doc = json.load(handle)
            self.sources = doc.get("sources", {})
            self.source_meta = {
                k: v for k, v in doc.items() if k != "sources"
            }

        for module_dir in sorted(d for d in self.root.iterdir() if d.is_dir()):
            for md_path in sorted(module_dir.glob("*.md")):
                try:
                    entry = self._load_entry(md_path, module_dir.name)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to load %s: %s", md_path, exc)
                    continue
                self.entries[(entry.module_id, entry.class_label)] = entry

        self.available = bool(self.entries)
        if self.available:
            logger.info(
                "Knowledge base loaded: %d entries, %d sources",
                len(self.entries),
                len(self.sources),
            )

        self._warn_unknown_citations()

    def _load_entry(self, path: Path, fallback_module: str) -> KnowledgeEntry:
        text = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(text)
        sections = _parse_sections(body)

        return KnowledgeEntry(
            class_label=str(meta.get("class", path.stem)),
            module_id=str(meta.get("module", fallback_module)),
            risk_tier=str(meta.get("risk_tier", "Unknown")),
            requires_clinical_review=bool(meta.get("requires_clinical_review", True)),
            sources=list(meta.get("sources", [])),
            sections=sections,
            path=str(path),
        )

    def _warn_unknown_citations(self) -> None:
        """Flag citations that do not resolve to the source registry.

        A dangling citation means a report would display a reference the reader
        cannot follow, which defeats the purpose of the evidence layer.
        """
        dangling: dict[str, list[str]] = {}
        for (module_id, label), entry in self.entries.items():
            missing = [c for c in entry.all_citations() if c not in self.sources]
            if missing:
                dangling[f"{module_id}/{label}"] = missing

        if dangling:
            logger.warning("Unresolved citation IDs in knowledge base: %s", dangling)

    def get(self, module_id: str, class_label: str) -> KnowledgeEntry | None:
        """Look up one entry.

        Returns ``None`` when the label is unknown. Callers must handle that
        rather than substituting another class's narrative -- silently falling
        back to "No Tumor" for an unrecognised label produced false
        reassurance in the previous implementation.
        """
        return self.entries.get((module_id, class_label))

    def labels(self, module_id: str) -> list[str]:
        """Known class labels for a module."""
        return sorted(
            label for (mod, label) in self.entries.keys() if mod == module_id
        )

    def resolve_sources(self, source_ids: list[str]) -> list[dict]:
        """Expand source IDs into full citation records.

        Unknown IDs are returned with ``resolved: False`` rather than dropped,
        so a broken reference is visible instead of silently disappearing.
        """
        out = []
        for source_id in source_ids:
            record = self.sources.get(source_id)
            if record is None:
                out.append({"id": source_id, "resolved": False})
            else:
                out.append({"id": source_id, "resolved": True, **record})
        return out

    def chunks(self, module_id: str | None = None) -> list[dict]:
        """Emit retrieval chunks, one per section.

        Section boundaries are used directly as chunk boundaries: they are
        already coherent units of roughly the right length, so no sliding
        window is needed.

        Returns:
            List of dicts with ``chunk_id``, ``module``, ``class``, ``section``,
            ``heading``, ``text`` and ``citations``.
        """
        out = []
        for (mod, label), entry in sorted(self.entries.items()):
            if module_id and mod != module_id:
                continue
            for key, section in entry.sections.items():
                out.append(
                    {
                        "chunk_id": f"{mod}/{label}/{key}".replace(" ", "_"),
                        "module": mod,
                        "class": label,
                        "section": key,
                        "heading": section["heading"],
                        "text": section["text"],
                        "citations": section["citations"],
                    }
                )
        return out

    def stats(self) -> dict:
        """Summary for health endpoints and diagnostics."""
        chunk_list = self.chunks()
        all_citations: set[str] = set()
        for entry in self.entries.values():
            all_citations.update(entry.all_citations())

        return {
            "available": self.available,
            "entries": len(self.entries),
            "modules": sorted({mod for (mod, _) in self.entries.keys()}),
            "chunks": len(chunk_list),
            "sources_registered": len(self.sources),
            "citations_used": len(all_citations),
            "unresolved_citations": sorted(
                c for c in all_citations if c not in self.sources
            ),
            "entries_pending_review": sorted(
                f"{mod}/{label}"
                for (mod, label), e in self.entries.items()
                if e.requires_clinical_review
            ),
        }


@lru_cache(maxsize=1)
def get_knowledge_base(root: str | None = None) -> KnowledgeBase:
    """Return the process-wide knowledge base singleton.

    Args:
        root: Knowledge directory. Defaults to ``<repo>/knowledge``, resolved
            from this file's location rather than the process CWD.
    """
    if root is None:
        root = str(Path(__file__).resolve().parents[4] / "knowledge")
    return KnowledgeBase(Path(root))
