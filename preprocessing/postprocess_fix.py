"""
postprocess_fix_v2.py
=====================
Updated post-processing pass. Fixes all issues found across both QA rounds.

CHANGES vs v1
-------------
FIX-1  CRLF now uses BINARY mode
       Root cause of failure: Python open(mode='r') auto-converts \\r\\n → \\n
       when reading on Linux, so the string-level replace saw nothing to fix.
       When the file was then re-zipped (on Windows or a Windows zip tool),
       \\r\\n was re-introduced. Fixed by reading/writing raw bytes and doing
       a byte-level b'\\r\\n' → b'\\n' replacement before any text processing.

NEW-5  Short paragraph noise filter
       2,050 paragraphs (9% of dataset) are under 10 words — fragments like
       '"', 'mpp', 'Designation', '92. Omitted', 'Chapter V of'.
       These are marginal labels, garbled headers, and TOC remnants that
       slipped through cleaning. They will confuse BART/T5. Filtered out.

RETAINED from v1
----------------
FIX-2  Oversized paragraph re-chunking (works correctly, kept as-is)
FIX-3  Bracket footnote residue removal (constitution files)
FIX-4  Letter-spaced text collapse (D i g i t a l → Digital)
"""

import os
import re
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR  = "dataset/fixed_text"    # ← your current fixed_text folder
OUTPUT_DIR = "dataset/ready_text"    # ← final model-ready output

MAX_PARA_WORDS = 400    # BART-large / T5 safe word cap
MIN_PARA_WORDS = 10     # paragraphs shorter than this are filtered as noise

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIX-1  CRLF — binary mode (the ONLY reliable approach)
#
# Python text mode on Linux silently converts \\r\\n → \\n on read, making
# string-level replacement a no-op. Binary mode sees the raw bytes.
# We strip \\r bytes entirely: b'\\r\\n' → b'\\n', stray b'\\r' → b'\\n'.
# ─────────────────────────────────────────────────────────────────────────────

def strip_crlf_bytes(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


# ─────────────────────────────────────────────────────────────────────────────
# FIX-3  Bracket footnote residue
#
# Patterns like  1[SOVEREIGN SOCIALIST...]  and  2[unity and integrity...]
# are amendment-insertion markers in Constitution files. We keep the text
# inside the brackets but strip the numeric marker prefix.
#
# Rule: N[real text] → real text   (only when content contains actual words)
# Also strip trailing superscript numbers attached to words: "Constitution1" → "Constitution"
# ─────────────────────────────────────────────────────────────────────────────

_BRACKET_FOOTNOTE_RE = re.compile(r"\b(\d{1,3})\[([^\]]{1,500})\]")

def _bracket_replace(m: re.Match) -> str:
    content = m.group(2).strip()
    if re.search(r"[A-Za-z]{2,}", content):
        return content
    return m.group(0)   # leave ambiguous cases (short or punctuation-only)

def remove_bracket_footnotes(text: str) -> str:
    text = _BRACKET_FOOTNOTE_RE.sub(_bracket_replace, text)
    # Strip stray superscript numbers attached to word endings
    text = re.sub(r"(?<=[a-z\.\"\'»\)])\d{1,2}(?=[\s,;:\.\"\'\n])", "", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# FIX-4  Letter-spaced text collapse
#
# "D i g i t a l   I n d i a" → "Digital India"
# Only collapses when all tokens are 1-2 chars and the run is 4+ tokens long.
# Conservative to avoid false-positives on normal text like "A B and C".
# ─────────────────────────────────────────────────────────────────────────────

_LETTERSPACED_RE = re.compile(
    r"(?<!\w)((?:[A-Za-z]{1,2} ){3,}[A-Za-z]{1,2})(?!\w)"
)

def _collapse_letterspaced(m: re.Match) -> str:
    run = m.group(1)
    tokens = run.split(" ")
    if all(1 <= len(t) <= 2 for t in tokens) and len(tokens) >= 4:
        return "".join(tokens)
    return run

def fix_letter_spacing(text: str) -> str:
    return _LETTERSPACED_RE.sub(_collapse_letterspaced, text)


# ─────────────────────────────────────────────────────────────────────────────
# FIX-2  Oversized paragraph re-chunking
#
# For any paragraph still over MAX_PARA_WORDS after splitting on \\n\\n,
# inject sentence-boundary breaks first (O(n) forward scan, no lookbehind),
# then re-chunk into ≤400-word pieces.
# ─────────────────────────────────────────────────────────────────────────────

_PROVISION_STARTERS = [
    r"BE\s+it\s+enacted",
    r"Provided\s+that",
    r"Provided\s+further\s+that",
    r"Provided\s+also\s+that",
    r"Explanation\.?(?:—|\s)",
    r"WHEREAS",
    r"NOW,?\s+THEREFORE",
    r"CHAPTER\s+[IVXLCDM]+",
    r"PART\s+[IVXLCDM]+",
    r"SCHEDULE\s+[IVXLCDM]*",
]

_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|i\.e|e\.g|"
    r"No|Sec|Art|Cl|cl|Ch|ss|sub|para|pt|vol|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"Ltd|Co|Corp|Inc|Pvt|Govt|Dept|Min|Div|"
    r"approx|est|ref|viz)\.",
    re.IGNORECASE,
)


def _inject_breaks(text: str) -> str:
    """Inject paragraph/sentence breaks into flat blobs. O(n), no lookbehind."""
    for starter in _PROVISION_STARTERS:
        text = re.sub(r"(?<!\n)(" + starter + r")", r"\n\n\1", text, flags=re.IGNORECASE)

    text = re.sub(r"(?<=\s)(\d{1,3})\.\s+([A-Z][a-z]{2,})", r"\n\n\1. \2", text)
    text = re.sub(r"\s+\((\d{1,3})\)\s+([A-Za-z])", r"\n(\1) \2", text)
    text = re.sub(r"\s+\(([a-z])\)\s+([A-Za-z])", r"\n(\1) \2", text)

    tokens = re.split(r"(\. )(?=[A-Z])", text)
    out: list[str] = []
    for i, tok in enumerate(tokens):
        if tok == ". " and i > 0:
            prev = "".join(out)
            last_m = re.search(r"(\b\w+)$", prev)
            is_abbrev = last_m and _ABBREV_RE.search(last_m.group(1) + ".")
            is_digit  = last_m and re.match(r"^\d+$", last_m.group(1))
            out.append(".\n" if not (is_abbrev or is_digit) else ". ")
        else:
            out.append(tok)
    text = "".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(ln.strip() for ln in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sentences_safe(text: str) -> list[str]:
    """O(n) sentence splitter — no regex lookbehind."""
    lines = text.split("\n")
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line.split()) <= MAX_PARA_WORDS:
            result.append(line)
            continue
        current_start = 0
        i = 0
        while i < len(line) - 2:
            if line[i] in ".!?" and line[i + 1] == " " and line[i + 2].isupper():
                chunk = line[current_start: i + 1].strip()
                if chunk:
                    result.append(chunk)
                current_start = i + 2
                i += 3
            else:
                i += 1
        tail = line[current_start:].strip()
        if tail:
            result.append(tail)
    return result


def _chunk_paragraph(para: str) -> list[str]:
    if len(para.split()) <= MAX_PARA_WORDS:
        return [para]
    sentences = _split_sentences_safe(para)
    chunks: list[str] = []
    current: list[str] = []
    current_wc = 0
    for sent in sentences:
        swc = len(sent.split())
        if swc > MAX_PARA_WORDS:
            if current:
                chunks.append(" ".join(current))
                current, current_wc = [], 0
            chunks.append(sent)
            continue
        if current_wc + swc > MAX_PARA_WORDS:
            chunks.append(" ".join(current))
            current, current_wc = [sent], swc
        else:
            current.append(sent)
            current_wc += swc
    if current:
        chunks.append(" ".join(current))
    return chunks if chunks else [para]


def rechunk(text: str) -> str:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    result: list[str] = []
    for para in paragraphs:
        if len(para.split()) > MAX_PARA_WORDS:
            para = _inject_breaks(para)
            sub_paras = [sp for sp in para.split("\n\n") if sp.strip()]
            for sp in sub_paras:
                result.extend(_chunk_paragraph(sp))
        else:
            result.extend(_chunk_paragraph(para))
    return "\n\n".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# NEW-5  Short paragraph noise filter
#
# Paragraphs under MIN_PARA_WORDS (10) are fragments that slipped through
# cleaning: section labels, garbled headers, lone punctuation, TOC remnants.
# Examples found: '"', 'mpp', 'Designation', '92. Omitted', 'Chapter V of'
# These will confuse the summariser — filter them out entirely.
# ─────────────────────────────────────────────────────────────────────────────

def filter_noise_paragraphs(text: str) -> tuple[str, int]:
    """
    Remove paragraphs under MIN_PARA_WORDS words.
    Returns (cleaned_text, number_of_paragraphs_removed).
    """
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    kept = []
    removed = 0
    for para in paragraphs:
        if len(para.split()) >= MIN_PARA_WORDS:
            kept.append(para)
        else:
            removed += 1
    return "\n\n".join(kept), removed


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel check
# ─────────────────────────────────────────────────────────────────────────────

def is_sentinel(text: str) -> bool:
    return text.strip().startswith("[EXTRACTION_FAILED")


# ─────────────────────────────────────────────────────────────────────────────
# Quality report
# ─────────────────────────────────────────────────────────────────────────────

def quality_report(text: str, fname: str, noise_removed: int) -> dict:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    word_counts = [len(p.split()) for p in paragraphs]
    return {
        "file"          : fname,
        "total_words"   : sum(word_counts),
        "paragraphs"    : len(paragraphs),
        "max_para_wc"   : max(word_counts) if word_counts else 0,
        "over_limit"    : sum(1 for wc in word_counts if wc > MAX_PARA_WORDS),
        "noise_removed" : noise_removed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def fix_document(raw_bytes: bytes, filename: str) -> tuple[str, dict]:
    """
    Apply all fixes in order. Accepts raw bytes, returns (fixed_text, report).

    Order matters:
      1. Binary CRLF strip (on bytes, before any text processing)
      2. Decode to string
      3. Bracket footnote cleanup
      4. Letter-spacing collapse
      5. Paragraph rechunking
      6. Noise paragraph filter (last, so we filter already-chunked output)
    """
    # FIX-1: binary CRLF strip
    raw_bytes = strip_crlf_bytes(raw_bytes)

    # Decode
    text = raw_bytes.decode("utf-8", errors="replace")

    # FIX-3: bracket footnotes
    text = remove_bracket_footnotes(text)

    # FIX-4: letter-spaced text
    text = fix_letter_spacing(text)

    # FIX-2: rechunk oversized paragraphs
    text = rechunk(text)

    # NEW-5: filter short noise paragraphs
    text, noise_removed = filter_noise_paragraphs(text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text, quality_report(text, filename, noise_removed)


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".txt"))

    if not files:
        print(f"No .txt files found in {INPUT_DIR}/")
        return

    skipped_sentinels: list[str] = []
    reports: list[dict] = []
    still_oversized: list[dict] = []
    total_noise_removed = 0

    for fname in tqdm(files, desc="Fixing"):
        fpath = os.path.join(INPUT_DIR, fname)

        # Read as RAW BYTES — critical for reliable CRLF stripping
        with open(fpath, "rb") as f:
            raw = f.read()

        # Sentinel check (decode just enough to check the header)
        if is_sentinel(raw.decode("utf-8", errors="replace")):
            skipped_sentinels.append(fname)
            continue

        fixed_text, report = fix_document(raw, fname)
        reports.append(report)
        total_noise_removed += report["noise_removed"]

        if report["over_limit"] > 0:
            still_oversized.append(report)

        # Write as UTF-8 with LF line endings (open with newline='\n' to prevent
        # Python from re-adding \r\n on any platform)
        out_path = os.path.join(OUTPUT_DIR, fname)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(fixed_text)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_words = sum(r["total_words"] for r in reports)
    total_paras = sum(r["paragraphs"] for r in reports)
    ready_paras = sum(
        1 for r in reports
        for _ in range(r["paragraphs"])
    )

    print(f"\n{'='*62}")
    print(f"  POST-PROCESSING v2 COMPLETE")
    print(f"{'='*62}")
    print(f"  Files processed      : {len(reports)}")
    print(f"  Files skipped (OCR)  : {len(skipped_sentinels)}")
    print(f"  Output               : {OUTPUT_DIR}/")

    print(f"\n  Dataset totals")
    print(f"  ├─ Total words             : {total_words:,}")
    print(f"  ├─ Total paragraphs        : {total_paras:,}")
    print(f"  ├─ Avg words / paragraph   : {total_words // max(total_paras, 1)}")
    print(f"  └─ Noise paragraphs removed: {total_noise_removed:,}")

    if still_oversized:
        print(f"\n  ⚠  {len(still_oversized)} files still have paragraphs > {MAX_PARA_WORDS}w")
        print(f"     (Likely pure table/financial data — consider excluding)")
        still_oversized.sort(key=lambda r: -r["max_para_wc"])
        for r in still_oversized[:5]:
            print(f"     max {r['max_para_wc']}w in {r['file'][:55]}")
    else:
        print(f"\n  ✓ All paragraphs within {MAX_PARA_WORDS}-word limit")

    if skipped_sentinels:
        print(f"\n  Skipped (need OCR re-extraction):")
        for s in skipped_sentinels:
            print(f"     {s}")
        print(f"\n  To fix, run:")
        print(f"     pip install pytesseract pdf2image")
        print(f"     python ocr_extract.py")

    print(f"\n  Your dataset is ready for the summariser.")
    print(f"  Next step → python summarizer.py --input {OUTPUT_DIR}/")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()