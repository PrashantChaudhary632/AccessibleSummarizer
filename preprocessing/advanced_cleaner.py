"""
advanced_cleaner_final.py
=========================
Final production cleaner. Resolves ALL issues found across three QA rounds:

  Round-1 bugs  → v2 (line endings, footnotes, chapter banners, boilerplate)
  Round-2 bugs  → v3 (CRLF, inline footnotes, running headers, marginal notes,
                       separators, empty-file guard)
  Round-3 bugs  → THIS FILE (gazette inline variant, oversized paragraphs,
                       wef footnote compact form, encoding residue)

New in this version
-------------------
FIX-A  Gazette inline header (J&K style)
       Pattern: "82 THE GAZETTE OF INDIA : EXTRAORDINARY [PART II—SEC. 3(ii)]"
       appears mid-line or as standalone — previous regex missed the numeric
       prefix and the colon variant. Now handled with two targeted patterns.

FIX-B  Paragraph chunking for summariser
       T5-small / BART-base accept max 512–1024 tokens (~400–700 words).
       Files like English_Annual_Report (avg 5607w/para) would be silently
       truncated by the model, losing most content. This pass splits any
       paragraph exceeding MAX_PARA_WORDS at the nearest sentence boundary.

FIX-C  Compact w.e.f. footnote form
       the_constitution_of_india.txt uses "(w.e.f.13-4-1978)" with NO space —
       the previous regex required a space after the dot. Now handled with
       a tighter pattern that matches both spaced and compact forms.

FIX-D  Bullet / pipe table row removal
       Report PDFs (RBI Annual Report, HDR) contain pdfplumber-extracted
       table rows like "· 2025  2024  Change  2023" that are noise.
       Removed with a bullet-prefixed numeric line pattern.

FIX-E  Repeated short header lines (report PDFs)
       Some PDFs repeat the document title on every page as a running header.
       Detected by finding lines that appear > 3 times identically, then
       removed if they are short (< 12 words) and all-caps or title-case.
"""

import os
import re
import unicodedata
from collections import Counter
from tqdm import tqdm

INPUT_DIR  = "dataset/raw_text"
OUTPUT_DIR = "dataset/cleaned_text"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Maximum words per paragraph before chunking (FIX-B)
# BART-large: ~1024 tokens ≈ 700 words safe limit
# T5-small:   ~512 tokens  ≈ 350 words safe limit
# We use 400 as a conservative default that works for both
MAX_PARA_WORDS = 400


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Line endings (must be first)
# ─────────────────────────────────────────────────────────────────────────────

def fix_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Unicode + encoding noise
# ─────────────────────────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def remove_encoding_noise(text: str) -> str:
    text = re.sub(r"M-[bB]M-\^[@-_]M-[^\s]{1,3}", " ", text)   # UTF-8 mojibake
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # control chars
    text = re.sub(r"[\ufffd]+", " ", text)                          # replacement chars
    text = re.sub(r"[\u0900-\u109f\u1c00-\u1cff\ua800-\ua82f]+", " ", text)  # Indic scripts
    text = re.sub(r"[^\x00-\x7f]{3,}", " ", text)                  # long non-ASCII runs
    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Boilerplate: gazette headers, ministry lines, print codes
# ─────────────────────────────────────────────────────────────────────────────

_BOILERPLATE_LINE_RE = re.compile(
    r"""^\s*(?:
        REGISTERED\s+NO\.?\s*DL
      | PUBLISHED\s+BY\s+AUTHORITY
      | EXTRAORDINARY\s*$
      | Published\s+By\s+Authority
      | Website\s*:\s*\S+
      | DOI\s+\S+
      | Ministry\s+of\s+[A-Z]
      | UPLOADED\s+BY\s+THE\s+MANAGER
      | AND\s+PUBLISHED\s+BY\s+THE\s+CONTROLLER
      | MGIPMRND
      | \[F\.\s*No\.
      | Under\s+Secretary\s+to\s+the\s+Government
      | Deputy\s+Secretary\s+to\s+the\s+Government
      | S\.O\.\s*\(E\)\.?\s*$
      | jftLV|vlk\/kkj|Hkkx\s*II|izkf\/kdkj\s+ls
    )""",
    re.VERBOSE | re.IGNORECASE,
)

# FIX-A: Gazette inline patterns — covers both standalone lines and mid-text occurrences
_GAZETTE_STANDALONE = re.compile(
    r"(?m)^\s*\d*\s*THE\s+GAZETTE\s+OF\s+INDIA\s*:?\s*"
    r"(?:EXTRAORDINARY\s*)?\[?[^\]\n]{0,60}\]?\s*$",
    re.IGNORECASE,
)
_GAZETTE_INLINE = re.compile(
    r"\d+\s+THE\s+GAZETTE\s+OF\s+INDIA\s*:?\s*"
    r"(?:EXTRAORDINARY\s*)?\[?[A-Z\s\-—]{0,50}\]?\s*",
    re.IGNORECASE,
)
_GAZETTE_BARE = re.compile(
    r"THE\s+GAZETTE\s+OF\s+INDIA\s+EXTRAORDINARY\s*\[PART[^\]\n]{0,60}\]?",
    re.IGNORECASE,
)

_BOILERPLATE_INLINE = [
    r"Uploaded\s+by\s+Dte\.\s+of\s+Printing[^\n]*",
    r"AND\s+\(S\d+\)—\d{2}-\d{2}-\d{4}[^\n]*",
    r"\[F\.\s*No\.[^\]]{0,80}\]",
    r"\(F\.\s*No\.[^\)]{0,80}\)",
    r"[jJ]ftLV[^\n]{0,80}",
]

def remove_boilerplate(text: str) -> str:
    # Line-by-line filter
    lines = [ln for ln in text.split("\n") if not _BOILERPLATE_LINE_RE.match(ln)]
    text  = "\n".join(lines)
    # Gazette patterns (FIX-A)
    text = _GAZETTE_STANDALONE.sub("", text)
    text = _GAZETTE_INLINE.sub(" ", text)
    text = _GAZETTE_BARE.sub("", text)
    # Other inline boilerplate
    for pat in _BOILERPLATE_INLINE:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Footnote removal (all formats)
# ─────────────────────────────────────────────────────────────────────────────

def remove_footnotes(text: str) -> str:
    # Format A: line-start numbered footnotes (CGST, J&K acts)
    text = re.sub(
        r"(?m)^\d{1,3}\s+(?:Substituted|Omitted|Inserted|Added|Renumbered|Replaced)"
        r".+?(?=\n\s*\n|\n\d{1,3}\s+(?:Substituted|Omitted|Inserted)|$)",
        "",
        text,
        flags=re.DOTALL,
    )
    # Format B: inline superscript prefix (Constitution)
    text = re.sub(
        r"\d{1,3}(?:Subs?|Ins|Art|Cl|Added|Omitted|Rep|Renumbered)\."
        r"[^.]{0,300}?(?:w\.e\.f\.?[^.\n]{0,80}\.?|ibid\.[^.\n]{0,80}\.?|\.\s)",
        "",
        text,
    )
    # Format C: asterisk-prefixed (Constitution pending amendments)
    text = re.sub(
        r"(?m)^\*{1,3}(?:Ins|Subs|Cl\.|Art\.|Sub-clause|Now|The\s+words)[^\n]+",
        "",
        text,
    )
    # Format D: bracket-enclosed amendment citations (IT Act, CrPC)
    text = re.sub(
        r"\[(?:Substituted|Inserted|Omitted|Amended)\s+by[^\]]{0,300}\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # FIX-C: compact w.e.f. form "(w.e.f.13-4-1978)" — no space after dot
    text = re.sub(
        r"\(?w\.e\.f\.?\s*[\d\-\.]+(?:\s*\w+)?(?:\s*\d{4})?\)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Abbreviation key lines (Constitution preamble)
    # Matches lines like: Subs. '' Substituted.  Ins. '' Inserted.
    # Uses escaped Unicode for the double-prime (\u2033) to avoid SyntaxError
    text = re.sub(
        r"(?m)^(?:Subs?\.|Ins\.|Sch\.|Cl\.|Art\.)\s+[\u2018\u2019\u201c\u201d\u2033\"].*$",
        "",
        text,
    )
    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Running page headers
# ─────────────────────────────────────────────────────────────────────────────

_RUNNING_HDRS = [
    re.compile(
        r"THE\s+CONSTITUTION\s+OF\s+INDIA\s+\d+\s*\(Part\s+[IVXLCDMivxlcdm]+[^\)]*\)",
        re.IGNORECASE,
    ),
    re.compile(r"(?m)^\s*\d{1,4}\s*$"),
    re.compile(r"(?m)^—\d+—\s*$"),
    re.compile(r"\[\s*II[—\-]\s*3\s*\(\s*ii\s*\)\s*\]\s*:\s*\d+"),
    re.compile(r"(?m)^\s*Page\s+\d+\s*$", re.IGNORECASE),
]

def remove_running_headers(text: str) -> str:
    for pat in _RUNNING_HDRS:
        text = pat.sub("", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Table of contents
# ─────────────────────────────────────────────────────────────────────────────

def remove_table_of_contents(text: str) -> str:
    # 3+ consecutive lines that end with a page number
    text = re.sub(r"(?m)(?:^[A-Z][A-Z\s]+\s+\d+\s*\n){3,}", "\n", text)
    for pat in [
        r"(?is)(?:Table\s+of\s+)?Contents\s*\n.{0,3000}?(?=\n\n)",
        r"(?is)Arrangement\s+of\s+Sections.{0,2000}?(?=\n\n)",
    ]:
        text = re.sub(pat, "\n", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Chapter/section banner deduplication
# ─────────────────────────────────────────────────────────────────────────────

_BANNER_RE = re.compile(
    r"^\s*(CHAPTER\s+[IVXLCDM]+(?:\s+[A-Z][A-Z\s]*)?|"
    r"SCHEDULE\s+[IVXLCDM]+|"
    r"PART\s+[IVXLCDM]+(?:\s+[A-Z][A-Z\s]*)?|"
    r"REMOVAL\s+OF\s+DIFFICULTY\s+ORDERS)\s*$",
    re.IGNORECASE,
)

def deduplicate_banners(text: str) -> str:
    seen: set = set()
    out = []
    for line in text.split("\n"):
        m = _BANNER_RE.match(line)
        if m:
            key = re.sub(r"\s+", " ", m.group(1)).upper().strip()
            if key in seen:
                continue
            seen.add(key)
        out.append(line)
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Separator lines and bullet table rows (FIX-D)
# ─────────────────────────────────────────────────────────────────────────────

def remove_separators_and_noise(text: str) -> str:
    # Separator lines
    text = re.sub(r"(?m)^[\-_=\*\.]{3,}\s*$", "", text)
    text = re.sub(r"[\-_=]{3,}", " ", text)
    text = re.sub(r"[—–]{2,}", "—", text)
    # FIX-D: bullet/dot-prefixed numeric table rows from PDF tables
    # e.g. "· 2025  2024  Change  2023  2022  2021"
    text = re.sub(r"(?m)^[·•]\s+(?:[\d,\.\s\|]+\s*){3,}$", "", text)
    # Lines that are purely numbers/symbols with no alphabetic content
    text = re.sub(r"(?m)^(?:[\d,\.\s\|\-/%]+){6,}$", "", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Repeated running title headers (FIX-E)
# ─────────────────────────────────────────────────────────────────────────────

def remove_repeated_headers(text: str) -> str:
    """
    FIX-E: Remove short lines that repeat identically 4+ times across the
    document — these are running page titles (e.g. 'Raiffeisen Bank International
    | Member of RBI Group | Financial Year 2025').
    """
    lines = text.split("\n")
    # Count occurrences of each non-empty line
    line_counts = Counter(ln.strip() for ln in lines if ln.strip())
    # Identify repeated headers: short, appears 4+ times
    repeated = {
        ln
        for ln, count in line_counts.items()
        if count >= 4 and len(ln.split()) <= 15
    }
    if not repeated:
        return text
    filtered = [ln for ln in lines if ln.strip() not in repeated]
    return "\n".join(filtered)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Marginal note removal
# ─────────────────────────────────────────────────────────────────────────────

_KNOWN_MARGINAL = [
    r"\bProtection\s+of\s+(?:life\s+and\s+personal\s+liberty|certain\s+rights)\b\.?",
    r"\bRight\s+to\s+(?:Freedom|education|Constitutional\s+Remedies)\b\.?",
    r"\bAbolition\s+of\s+(?:Untouchability|titles)\b\.?",
    r"\bEquality\s+of\s+opportunity[^.]{0,40}\.",
    r"\bProhibition\s+of\s+(?:discrimination|traffic)[^.]{0,40}\.",
    r"\bFreedom\s+(?:as\s+to|of\s+conscience)[^.]{0,40}\.",
    r"\b(?:Organisation|Promotion|Duty)\s+of\s+[A-Za-z\s]{5,30}\.",
    r"\b(?:Admission|Formation)\s+(?:or\s+)?(?:establishment\s+)?of\s+[A-Za-z\s]{5,30}\.",
    r"\bSaving\s+of\s+(?:laws|Certain\s+Laws)[^.]{0,40}\.",
    r"\bValidation\s+of\s+certain\s+Acts[^.]{0,40}\.",
]

def remove_marginal_notes(text: str) -> str:
    for pat in _KNOWN_MARGINAL:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    # Generic: isolated Title Case phrase ending in period between body sentences
    text = re.sub(
        r"(?<=[a-z,;])\s{2,}(?:[A-Z][a-z]+\s+){1,4}[A-Z][a-z]+\.\s+(?=[A-Z0-9\(])",
        " ",
        text,
    )
    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Paragraph structure repair
# ─────────────────────────────────────────────────────────────────────────────

_CONTINUATION = re.compile(
    r"^(?:[a-z\(\[\"\'\-]|and\s|or\s|but\s|of\s|to\s|in\s|for\s)"
)

def repair_structure(text: str) -> str:
    """Join genuine continuation lines; preserve blank-line paragraph breaks."""
    paragraphs = re.split(r"\n{2,}", text)
    out = []
    for para in paragraphs:
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if not lines:
            continue
        merged = lines[0]
        for line in lines[1:]:
            if _CONTINUATION.match(line):
                merged = merged.rstrip() + " " + line
            else:
                merged = merged + "\n" + line
        merged = re.sub(r"[ \t]{2,}", " ", merged)
        out.append(merged.strip())
    return "\n\n".join(p for p in out if p)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — Miscellaneous normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalize_misc(text: str) -> str:
    text = re.sub(r"\[\*+\.?\]", "", text)                          # [*****] markers
    text = re.sub(r"([\]\)\.])\d{1,3}(?=\s|[A-Z,;])", r"\1", text) # superscript refs
    text = re.sub(r"(?m)^(?:\s*[\d,\.]+\s+){4,}$", "", text)        # numeric table rows
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 13 — Paragraph chunking for summariser (FIX-B)
# ─────────────────────────────────────────────────────────────────────────────
#
# ROOT CAUSE OF HANG: the previous version used re.split(r"(?<=[.!?])\s+(?=[A-Z\(])")
# on paragraphs that contain thousands of punctuation characters (e.g. WHO statistics
# tables extracted as one giant paragraph with 2292 dots). Lookbehind + lookahead
# anchors over such input cause catastrophic backtracking — the regex engine never
# returns.
#
# FIX: Split on newlines first (which are already present in the text and are
# the natural unit boundary from pdfplumber). Only fall back to a simple
# ". Capital" string search — NO regex lookbehind — on the individual lines.
# This is O(n) instead of O(n^2).

def _split_sentences_safe(text: str) -> list[str]:
    """
    Split text into sentence-like units without using regex lookbehind.
    Strategy:
      1. Split on existing newlines first (cheapest, most reliable).
      2. For any remaining line longer than MAX_PARA_WORDS words, split at
         '. ' + uppercase boundaries using a simple forward scan.
    Never uses re.split with lookbehind — avoids catastrophic backtracking.
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line.split()) <= MAX_PARA_WORDS:
            result.append(line)
            continue
        # Forward scan for sentence boundaries: ". X" where X is uppercase
        current_start = 0
        i = 0
        while i < len(line) - 2:
            if line[i] in ".!?" and line[i + 1] == " " and line[i + 2].isupper():
                chunk = line[current_start:i + 1].strip()
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


def chunk_large_paragraph(para: str, max_words: int = MAX_PARA_WORDS) -> list[str]:
    """
    Split a paragraph exceeding max_words into smaller chunks.
    Uses safe O(n) sentence splitting — no regex lookbehind.
    """
    if len(para.split()) <= max_words:
        return [para]

    sentences = _split_sentences_safe(para)
    chunks: list[str] = []
    current: list[str] = []
    current_wc = 0

    for sent in sentences:
        sent_wc = len(sent.split())
        # Hard-limit: if a single sentence exceeds max_words, emit it alone
        if sent_wc > max_words:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_wc = 0
            chunks.append(sent)
            continue
        if current_wc + sent_wc > max_words:
            chunks.append(" ".join(current))
            current = [sent]
            current_wc = sent_wc
        else:
            current.append(sent)
            current_wc += sent_wc

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [para]


def chunk_all_paragraphs(text: str) -> str:
    """Apply chunk_large_paragraph to every paragraph in the document."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    result: list[str] = []
    for para in paragraphs:
        result.extend(chunk_large_paragraph(para))
    return "\n\n".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 14 — Empty-file guard
# ─────────────────────────────────────────────────────────────────────────────

def check_content(text: str, filename: str) -> str:
    if len(text.split()) < 30:
        return (
            f"[EXTRACTION_FAILED: {filename}]\n"
            "Source PDF appears to be a scanned image without embedded text.\n"
            "Re-extract using an OCR pipeline (pytesseract + pdf2image).\n"
        )
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def clean_document(text: str, filename: str = "") -> str:
    text = fix_line_endings(text)           # 1. CRLF
    text = normalize_unicode(text)          # 2. Unicode
    text = remove_encoding_noise(text)      # 2. Encoding artifacts
    text = remove_boilerplate(text)         # 3. Gazette headers etc (FIX-A)
    text = remove_footnotes(text)           # 4. All footnote formats (FIX-C)
    text = remove_running_headers(text)     # 5. Page running headers
    text = remove_table_of_contents(text)   # 6. TOC blocks
    text = deduplicate_banners(text)        # 7. Repeated chapter banners
    text = remove_separators_and_noise(text)# 8. Separators + table rows (FIX-D)
    text = remove_repeated_headers(text)    # 9. Repeated title lines (FIX-E)
    text = remove_marginal_notes(text)      # 10. Marginal notes
    text = repair_structure(text)           # 11. Paragraph structure
    text = normalize_misc(text)             # 12. Final normalisation
    text = chunk_all_paragraphs(text)       # 13. Summariser chunking (FIX-B)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    text = check_content(text, filename)    # 14. Empty-file guard
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".txt"))

    for fname in tqdm(files, desc="Cleaning"):
        fpath = os.path.join(INPUT_DIR, fname)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        cleaned = clean_document(raw, filename=fname)

        out_path = os.path.join(OUTPUT_DIR, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

    print(f"\nDone — {len(files)} files cleaned.")
    print(f"Output → {OUTPUT_DIR}/")