"""Author grounded clinical narratives offline, for human review.

The idea
--------
The narrative output space is small and enumerable: 4 classes x 3 uncertainty
bands = **12 narratives**. Everything else in a report (probabilities,
confidence, prediction set) is numeric slot-filling.

So rather than call an LLM per request, each of the 12 is generated **once**,
grounded in passages retrieved from the local cited knowledge base, then
**reviewed by a human** and committed. Serving reads the vetted text.

That buys, all at once: zero runtime latency, no API dependency during a demo,
no rate limits, and — the point that matters clinically — every sentence a user
sees has been read by a person first.

Privacy
-------
The prompt contains only the class label, the uncertainty band and passages from
the local knowledge base. **No image data and no patient identifiers are sent.**

Grounding check
---------------
Generated text may only cite source IDs that were present in the supplied
evidence. Any other ID is a fabricated citation and is reported per item as
``hallucinated_citations``, which is a measurable groundedness metric rather
than a vague assurance.

Usage
-----
Run from ``backend/fastapi``::

    python scripts/author_narratives.py --list-models
    python scripts/author_narratives.py --dry-run
    python scripts/author_narratives.py
    python scripts/author_narratives.py --classes Glioma --bands confident --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.services.knowledge import get_knowledge_base  # noqa: E402
from app.services.llm import generate, health, is_configured  # noqa: E402
from app.services.retrieval import get_retriever  # noqa: E402

CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9\-\.]*)\]")

# Models sometimes emit full-width CJK brackets instead of ASCII ones. Observed
# in practice: gpt-oss-120b produced 【ncbi-meningioma-statpearls】, which the
# citation regex missed entirely and reported as an ungrounded narrative -- a
# false negative on correct output. Normalise before validating, and instruct
# the model explicitly (see rule 2 in SYSTEM_PROMPT).
_BRACKET_TRANSLATION = str.maketrans({"【": "[", "】": "]", "〔": "[", "〕": "]"})

# Typographic characters the model emits freely that cause trouble downstream:
# U+2011 (non-breaking hyphen) crashed console output under Windows cp1252, and
# the default reportlab fonts do not cover several of these, so a PDF would show
# black boxes. Normalising to ASCII loses nothing meaningful here.
_PUNCTUATION_TRANSLATION = str.maketrans({
    "\u2010": "-",   # hyphen
    "\u2011": "-",   # non-breaking hyphen
    "\u2012": "-",   # figure dash
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2015": "-",   # horizontal bar
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u201e": '"',
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",   # non-breaking space
    "\u2009": " ",   # thin space
    "\u202f": " ",   # narrow no-break space
    "\u2212": "-",   # minus sign
})


def normalise_citations(text: str) -> str:
    """Normalise citation brackets and typographic punctuation to ASCII.

    Two separate problems, fixed together because both stem from the model's
    free choice of Unicode characters:

    1. Full-width CJK brackets around citations, which the citation regex misses.
    2. Smart quotes, dashes and non-breaking spaces, which break cp1252 console
       output and the default PDF fonts.
    """
    return text.translate(_BRACKET_TRANSLATION).translate(_PUNCTUATION_TRANSLATION)

BANDS = ["confident", "borderline", "indeterminate"]

SYSTEM_PROMPT = """You write clinical decision-support narratives for an AI medical imaging \
research system. You are NOT a diagnostician and the system is NOT a diagnostic device.

Absolute rules:
1. Use ONLY the EVIDENCE passages supplied. If the evidence does not cover \
something, omit it. Never add facts from your own knowledge.
2. Cite every clinical claim with a source ID in PLAIN ASCII square brackets, \
exactly like [nci-cns-pdq-hp]. Use the characters [ and ] only - never full-width \
or CJK brackets such as the ones used in Chinese or Japanese typesetting. Use \
ONLY IDs that appear in the supplied evidence.
3. Never name a medicine and never give a dose, route or schedule.
4. Never state or imply a confirmed diagnosis. The system produces a \
classification, not a diagnosis.
5. Do NOT invent specific numbers, percentages, sizes or timeframes. Refer to \
confidence qualitatively; exact figures are inserted separately by the report \
generator.
6. Phrase investigations as matters to discuss with a qualified healthcare \
professional, never as instructions to the reader.
7. Write in plain professional British English prose. No headings, no bullet \
lists, no markdown emphasis. 120-200 words.
8. Use only plain ASCII punctuation: straight quotes, ordinary hyphens, ordinary \
spaces. Do not use smart quotes, en or em dashes, non-breaking hyphens or \
ellipsis characters.
"""

BAND_FRAMING = {
    "confident": (
        "The model's conformal prediction set contains this single class, so the "
        "result is reported as the primary finding. Explain what the finding "
        "generally represents, state clearly that it is a classification rather "
        "than a diagnosis, and indicate the kinds of further evaluation that "
        "would normally be discussed with a clinician."
    ),
    "borderline": (
        "The model's conformal prediction set contains TWO classes, so the result "
        "is genuinely ambiguous between this class and one alternative. Do not "
        "present this class as the answer. Explain that the system cannot "
        "separate the candidates, why such confusion is plausible, and that "
        "professional review is the appropriate next step."
    ),
    "indeterminate": (
        "The model's conformal prediction set contains three or more classes, so "
        "the system has effectively failed to discriminate. Withhold any "
        "interpretation of this class. State plainly that the result is "
        "inconclusive, must not be acted upon, and requires review by a "
        "qualified radiologist or specialist physician."
    ),
}


def build_prompt(class_label: str, band: str, evidence: list) -> tuple[str, list[str]]:
    """Assemble the grounded prompt.

    Returns:
        ``(prompt, allowed_citation_ids)``
    """
    allowed: set[str] = set()
    blocks = []

    for hit in evidence:
        allowed.update(hit.citations)
        cites = ", ".join(hit.citations) if hit.citations else "none"
        blocks.append(
            f"--- EVIDENCE [{hit.section}] (source IDs: {cites}) ---\n{hit.text}"
        )

    prompt = f"""CLASSIFICATION CONTEXT
Modality: Brain MRI
Model output class: {class_label}
Uncertainty band: {band}

SITUATION
{BAND_FRAMING[band]}

PERMITTED SOURCE IDS (cite only these)
{", ".join(sorted(allowed)) if allowed else "none"}

EVIDENCE
{chr(10).join(blocks)}

TASK
Write the narrative for this class and uncertainty band, following every rule in
your instructions. Output the prose only."""

    return prompt, sorted(allowed)


def check_grounding(text: str, allowed: list[str]) -> dict:
    """Validate that the text cites only permitted source IDs.

    Operates on bracket-normalised text so a typographic variant is not
    mistaken for a missing citation.
    """
    text = normalise_citations(text)
    used = sorted(set(CITATION_RE.findall(text)))
    allowed_set = set(allowed)
    hallucinated = [c for c in used if c not in allowed_set]

    # Crude but useful guards against the rules most likely to be broken.
    numbers = re.findall(r"\b\d+(?:\.\d+)?\s?%", text)

    # Anything outside ASCII after normalisation will still break cp1252
    # consoles and the default PDF fonts, so surface it rather than discover it
    # at render time.
    non_ascii = sorted({ch for ch in text if ord(ch) > 127})

    return {
        "citations_used": used,
        "citations_allowed": allowed,
        "hallucinated_citations": hallucinated,
        "grounded": not hallucinated and bool(used),
        "uncited": not used,
        "contains_percentages": numbers,
        "non_ascii_characters": [f"U+{ord(c):04X}" for c in non_ascii],
        "ascii_clean": not non_ascii,
        "word_count": len(text.split()),
    }


def list_models() -> int:
    """Print the models available to the configured key."""
    if not is_configured():
        print("GROQ_API is not set in the repo-root .env", file=sys.stderr)
        return 2
    try:
        from groq import Groq

        client = Groq(api_key=settings.GROQ_API)
        for model in sorted(client.models.list().data, key=lambda m: m.id):
            mark = "  <- configured" if model.id == settings.GROQ_MODEL else ""
            print(f"  {model.id}{mark}")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not list models: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompts and show evidence, but call no API")
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--bands", nargs="*", default=None, choices=BANDS)
    parser.add_argument("--k", type=int, default=5, help="Evidence chunks per narrative")
    parser.add_argument(
        "--grounding-attempts",
        type=int,
        default=3,
        help="Max attempts per narrative to satisfy the citation rules",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--module", default="brain_mri")
    args = parser.parse_args()

    if args.list_models:
        return list_models()

    kb = get_knowledge_base()
    if not kb.available:
        print("ERROR: knowledge base unavailable.", file=sys.stderr)
        return 2

    classes = args.classes or kb.labels(args.module)
    bands = args.bands or BANDS
    out_dir = Path(kb.root) / "narratives"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== configuration ===")
    for key, value in health().items():
        print(f"  {key:12s} {value}")
    if args.model:
        print(f"  {'override':12s} {args.model}")
    print(f"  {'classes':12s} {classes}")
    print(f"  {'bands':12s} {bands}")
    print(f"  {'output':12s} {out_dir}")
    print(f"  {'total items':12s} {len(classes) * len(bands)}")

    if not args.dry_run and not is_configured():
        print("\nERROR: GROQ_API not set. Use --dry-run to inspect prompts.", file=sys.stderr)
        return 2

    retriever = get_retriever("auto")
    print(f"  {'retriever':12s} {retriever.backend}")

    summary = []
    written = 0
    skipped = 0
    failed = 0

    for class_label in classes:
        entry = kb.get(args.module, class_label)
        if entry is None:
            print(f"\n[{class_label}] no knowledge entry — skipping", file=sys.stderr)
            continue

        for band in bands:
            slug = f"{class_label.lower().replace(' ', '_')}_{band}"
            path = out_dir / f"{slug}.json"

            if path.exists() and not args.force:
                print(f"\n[{slug}] exists — skipping (use --force to regenerate)")
                skipped += 1
                continue

            # Class-filtered retrieval: evidence must come from this class's
            # own sections, not another class's similar-sounding text.
            evidence = retriever.search(
                f"{class_label} overview, investigations, follow-up, warning signs, "
                "treatment information and AI limitations",
                k=args.k,
                module=args.module,
                class_label=class_label,
            )

            prompt, allowed = build_prompt(class_label, band, evidence)

            print(f"\n[{slug}] evidence: {len(evidence)} chunks, "
                  f"{len(allowed)} permitted source IDs")
            for hit in evidence:
                print(f"    {hit.section:26s} score={hit.score:.4f}")

            if args.dry_run:
                print(f"    prompt: {len(prompt.split())} words (dry run, no API call)")
                continue

            # Generate, then re-attempt if the grounding rules were broken.
            # Roughly 1 in 12 first-pass outputs omits citations entirely, which
            # is unusable for an evidence-grounded report. Both the first-pass
            # and final rates are recorded so the LLM layer's reliability is
            # measured rather than hidden by the retry.
            result = None
            narrative_text = ""
            grounding: dict = {}
            first_pass_grounded = None
            attempts_used = 0

            for attempt in range(1, args.grounding_attempts + 1):
                attempt_prompt = prompt
                if attempt > 1:
                    attempt_prompt = (
                        prompt
                        + "\n\nCORRECTION REQUIRED\nThe previous attempt broke a "
                        "rule. Every clinical claim MUST end with a source ID in "
                        "plain ASCII square brackets, drawn only from the "
                        "permitted list above, for example [nci-meningioma]. "
                        "Rewrite the narrative with those citations present."
                    )

                result = generate(attempt_prompt, system=SYSTEM_PROMPT, model=args.model)
                attempts_used = attempt

                if not result.ok:
                    break

                narrative_text = normalise_citations(result.text)
                grounding = check_grounding(narrative_text, allowed)

                if first_pass_grounded is None:
                    first_pass_grounded = grounding["grounded"]

                if grounding["grounded"]:
                    break

                print(
                    f"    attempt {attempt} not grounded "
                    f"(uncited={grounding['uncited']}, "
                    f"hallucinated={grounding['hallucinated_citations']}) — retrying"
                )

            if result is None or not result.ok:
                error = result.error if result else "no attempt made"
                print(f"    FAILED: {error}", file=sys.stderr)
                failed += 1
                summary.append({"slug": slug, "ok": False, "error": error})
                continue

            doc = {
                "class": class_label,
                "module": args.module,
                "band": band,
                "text": narrative_text,
                "grounding": grounding,
                "grounding_attempts_used": attempts_used,
                "grounded_on_first_attempt": first_pass_grounded,
                "evidence_chunks": [h.chunk_id for h in evidence],
                "llm": result.to_dict(),
                "generated": datetime.now(timezone.utc).isoformat(),
                # The gate that matters. Serving must refuse unreviewed text.
                "reviewed": False,
                "reviewed_by": None,
                "reviewed_at": None,
                "review_instructions": (
                    "Check every sentence against the cited sources. Confirm no "
                    "medicine, dose or diagnosis is stated and that the "
                    "uncertainty framing matches the band. Then set "
                    "reviewed=true and fill reviewed_by/reviewed_at."
                ),
            }

            with open(path, "w", encoding="utf-8") as handle:
                json.dump(doc, handle, indent=2)

            flags = []
            if grounding["hallucinated_citations"]:
                flags.append(f"HALLUCINATED {grounding['hallucinated_citations']}")
            if grounding["uncited"]:
                flags.append("NO CITATIONS")
            if grounding["contains_percentages"]:
                flags.append(f"PERCENTAGES {grounding['contains_percentages']}")
            if not grounding["ascii_clean"]:
                flags.append(f"NON-ASCII {grounding['non_ascii_characters']}")

            status = "  ".join(flags) if flags else "clean"
            print(f"    {grounding['word_count']} words, "
                  f"cites {grounding['citations_used']} -> {status}")
            print(f"    wrote {path.name}")

            written += 1
            summary.append({
                "slug": slug,
                "ok": True,
                "grounded": grounding["grounded"],
                "word_count": grounding["word_count"],
                "hallucinated": grounding["hallucinated_citations"],
                "percentages": grounding["contains_percentages"],
                "ascii_clean": grounding["ascii_clean"],
                "non_ascii": grounding["non_ascii_characters"],
                "attempts_used": attempts_used,
                "grounded_first_attempt": first_pass_grounded,
                "tokens": result.usage.get("total_tokens"),
            })

    # --- summary -----------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"written {written} | skipped {skipped} | failed {failed}")

    if summary:
        ok_items = [s for s in summary if s.get("ok")]
        grounded = [s for s in ok_items if s.get("grounded")]
        with_halluc = [s for s in ok_items if s.get("hallucinated")]
        with_pct = [s for s in ok_items if s.get("percentages")]
        tokens = sum(s.get("tokens") or 0 for s in ok_items)

        not_ascii = [s for s in ok_items if not s.get("ascii_clean")]
        first_ok = [s for s in ok_items if s.get("grounded_first_attempt")]
        retried = [s for s in ok_items if (s.get("attempts_used") or 1) > 1]

        print(f"grounded on FIRST attempt          : {len(first_ok)}/{len(ok_items)}")
        print(f"grounded FINAL (after retries)     : {len(grounded)}/{len(ok_items)}")
        print(f"items needing a retry              : {len(retried)}")
        print(f"fabricated citations               : {len(with_halluc)}")
        print(f"contain percentages (rule breach)  : {len(with_pct)}")
        print(f"non-ASCII after normalisation      : {len(not_ascii)}")
        print(f"total tokens used                  : {tokens}")

        index_path = out_dir / "authoring_summary.json"
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "generated": datetime.now(timezone.utc).isoformat(),
                    "model": args.model or settings.GROQ_MODEL,
                    "retriever": retriever.backend,
                    "k": args.k,
                    "written": written,
                    "skipped": skipped,
                    "failed": failed,
                    "grounded_final": len(grounded),
                    "grounded_first_attempt": len(first_ok),
                    "needed_retry": len(retried),
                    "total_tokens": tokens,
                    "grounding_attempts_allowed": args.grounding_attempts,
                    "items": summary,
                },
                handle,
                indent=2,
            )
        print(f"\nWritten: {index_path}")

    if written:
        print(
            "\nNEXT STEP (required): a human must review each narrative and set "
            "reviewed=true. Serving ignores unreviewed text and falls back to the "
            "template report."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
