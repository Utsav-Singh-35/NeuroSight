"""Vetted clinical narratives for report generation.

The review gate
---------------
Narratives are authored offline by an LLM grounded in the local knowledge base
(see ``scripts/author_narratives.py``), then **reviewed by a human**. This module
serves only entries whose ``reviewed`` flag is ``true``.

That gate is the whole point. An unreviewed narrative is treated exactly like a
missing one: :meth:`NarrativeStore.get` returns ``None`` and the report falls
back to deterministic template text. So the failure mode of the LLM layer is
"less polished prose", never "unvetted clinical claims shown to a user".

No LLM call happens at request time. The output space is 4 classes x 3
uncertainty bands = 12 items, all precomputed, so serving cost is a dict lookup.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

VALID_BANDS = {"confident", "borderline", "indeterminate"}


@dataclass
class Narrative:
    """One authored narrative and its provenance."""

    class_label: str
    module_id: str
    band: str
    text: str
    citations: list[str]
    reviewed: bool
    reviewed_by: str | None
    reviewed_at: str | None
    model: str
    generated: str
    grounded: bool
    path: str

    def to_dict(self, include_text: bool = True) -> dict:
        out = {
            "class": self.class_label,
            "module": self.module_id,
            "band": self.band,
            "citations": self.citations,
            "reviewed": self.reviewed,
            "reviewed_by": self.reviewed_by,
            "authored_by_model": self.model,
            "grounded": self.grounded,
        }
        if include_text:
            out["text"] = self.text
        return out


class NarrativeStore:
    """Loads authored narratives and enforces the review gate."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.all: dict[tuple[str, str, str], Narrative] = {}
        self.available = False
        self._load()

    def _load(self) -> None:
        if not self.root.is_dir():
            logger.info("No narratives directory at %s; using template reports.", self.root)
            return

        for path in sorted(self.root.glob("*.json")):
            if path.name == "authoring_summary.json":
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    doc = json.load(handle)
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not read narrative %s: %s", path.name, exc)
                continue

            band = str(doc.get("band", ""))
            if band not in VALID_BANDS:
                logger.error(
                    "Narrative %s has invalid band %r; ignoring.", path.name, band
                )
                continue

            grounding = doc.get("grounding", {})
            narrative = Narrative(
                class_label=str(doc.get("class", "")),
                module_id=str(doc.get("module", "")),
                band=band,
                text=str(doc.get("text", "")).strip(),
                citations=list(grounding.get("citations_used", [])),
                reviewed=bool(doc.get("reviewed", False)),
                reviewed_by=doc.get("reviewed_by"),
                reviewed_at=doc.get("reviewed_at"),
                model=str(doc.get("llm", {}).get("model", "unknown")),
                generated=str(doc.get("generated", "")),
                grounded=bool(grounding.get("grounded", False)),
                path=str(path),
            )

            key = (narrative.module_id, narrative.class_label, narrative.band)
            self.all[key] = narrative

        self.available = bool(self.all)
        reviewed = sum(1 for n in self.all.values() if n.reviewed)

        if self.available:
            logger.info(
                "Narratives loaded: %d total, %d reviewed and servable, "
                "%d awaiting review.",
                len(self.all),
                reviewed,
                len(self.all) - reviewed,
            )
            if reviewed == 0:
                logger.warning(
                    "No narrative has been reviewed yet, so all reports will use "
                    "template text. Review the files in %s and set "
                    "reviewed=true to enable them.",
                    self.root,
                )

    def get(self, module_id: str, class_label: str, band: str) -> Narrative | None:
        """Return a narrative only if it exists **and** has been reviewed.

        Unreviewed narratives are deliberately withheld: showing unvetted
        clinical prose is worse than showing plainer template text.
        """
        narrative = self.all.get((module_id, class_label, band))
        if narrative is None:
            return None
        if not narrative.reviewed:
            logger.debug(
                "Narrative %s/%s/%s exists but is unreviewed; withholding.",
                module_id,
                class_label,
                band,
            )
            return None
        if not narrative.text:
            return None
        return narrative

    def stats(self) -> dict:
        """Review-status summary for health endpoints."""
        reviewed = [n for n in self.all.values() if n.reviewed]
        pending = [n for n in self.all.values() if not n.reviewed]
        ungrounded = [n for n in self.all.values() if not n.grounded]

        return {
            "available": self.available,
            "total": len(self.all),
            "reviewed_and_servable": len(reviewed),
            "awaiting_review": len(pending),
            "ungrounded": len(ungrounded),
            "modules": sorted({n.module_id for n in self.all.values()}),
            "pending_items": sorted(
                f"{n.class_label}/{n.band}" for n in pending
            ),
            "authored_by": sorted({n.model for n in self.all.values()}),
        }


@lru_cache(maxsize=1)
def get_narrative_store(root: str | None = None) -> NarrativeStore:
    """Process-wide narrative store singleton.

    Args:
        root: Narratives directory. Defaults to ``<repo>/knowledge/narratives``,
            resolved from this file rather than the process CWD.
    """
    if root is None:
        root = str(Path(__file__).resolve().parents[4] / "knowledge" / "narratives")
    return NarrativeStore(Path(root))
