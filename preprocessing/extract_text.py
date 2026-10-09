"""
extract_text_v2.py
==================
Improved PDF text extractor for legal / government documents.

ROOT CAUSES fixed vs original extract_text.py
----------------------------------------------
ROOT-1  Single-line output  (most critical)
        CAUSE: page.extract_text() with default y_tolerance=3 clusters lines
        that are < 3 pt apart into one long string with no newlines. Legal PDFs
        printed at 8–10pt with tight spacing trigger this constantly.
        FIX-A: Use word-level extraction (extract_words) which gives us each
               word's y-coordinate (top / bottom). We then reconstruct line and
               paragraph boundaries by clustering words by their y-position,
               using a dynamic line-spacing threshold computed per page.
        FIX-B: For any page whose text still comes out as ≤ 2 lines after
               word-based extraction, apply heuristic sentence/section splitting
               as a last-resort fallback.

ROOT-2  No newline between pages
        CAUSE: original code did `text += page_text + "\n"` — only one newline
        between pages, so pdfplumber's final line on page N merges with the
        first line on page N+1 after cleaning.
        FIX: Separate pages with "\n\n" (paragraph break).

ROOT-3  Scanned / image-only PDFs produce empty files silently
        CAUSE: page.extract_text() returns None or "" for image-only pages.
        The original code just appended nothing, writing a 0-byte output file
        with no warning. Downstream summarisation then silently received nothing.
        FIX: Count pages that yielded text. If a document yields < 20 words
        total, write a clear sentinel file and print a warning instead of an
        empty file. Log which files need OCR.

ROOT-4  Two-column / marginal-note layout mangled
        CAUSE: pdfplumber reads characters left-to-right across the full page
        width, so it interleaves column A and column B text within each line.
        FIX: Detect two-column layouts by checking whether words cluster into
        two distinct x-ranges. If so, extract each column separately using
        crop(), then concatenate them top-to-bottom.

ROOT-5  Hindi / bilingual gazette headers extracted as noise
        CAUSE: pdfplumber faithfully extracts all text including Devanagari
        headers. This is correct behaviour; the cleaner must handle it.
        FIX: No change in extractor — handled by advanced_cleaner_v3.py.

ROOT-6  File naming collision for nested subdirectories
        CAUSE: original code used `file` (just the filename, no path context),
        so two PDFs named "Act.pdf" in different subfolders would overwrite
        each other in the output.
        FIX: Preserve the relative subdirectory structure in the output path.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional

import pdfplumber
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FOLDER  = "documents"
OUTPUT_FOLDER = "dataset/raw_text"

# Words per document below this → treat as scanned / OCR failure
MIN_WORDS_THRESHOLD = 20

# y-distance (points) between word-tops beyond which we consider a new paragraph
# 1.8× the median line height works well for legal docs (catches both tight and
# normal spacing without merging across section breaks)
PARAGRAPH_GAP_MULTIPLIER = 1.8

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core: word-based text reconstruction with dynamic line detection
# ─────────────────────────────────────────────────────────────────────────────

def words_to_structured_text(words: list) -> str:
    """
    Given a list of pdfplumber word dicts (each has 'text', 'top', 'bottom',
    'x0', 'x1'), reconstruct text with proper line breaks and paragraph gaps.

    Algorithm:
    1. Sort words by (top, x0) — reading order.
    2. Compute median word height as a proxy for line height.
    3. Cluster words into lines: words whose 'top' values differ by less than
       half a line height belong to the same line.
    4. Detect paragraph gaps: vertical distance between consecutive lines that
       exceeds PARAGRAPH_GAP_MULTIPLIER × median_line_height.
    5. Within a line, join words with a space. Between lines use '\n'. Between
       paragraphs use '\n\n'.
    """
    if not words:
        return ""

    # Sort into reading order
    words_sorted = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))

    # Compute median word/line height
    heights = sorted(w["bottom"] - w["top"] for w in words_sorted)
    median_height = heights[len(heights) // 2] if heights else 10.0
    line_cluster_tolerance = median_height * 0.6  # within 60% of line height → same line
    paragraph_gap = median_height * PARAGRAPH_GAP_MULTIPLIER

    # Cluster words into lines
    lines: list[list[dict]] = []
    current_line: list[dict] = [words_sorted[0]]
    current_top = words_sorted[0]["top"]

    for word in words_sorted[1:]:
        if abs(word["top"] - current_top) <= line_cluster_tolerance:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]
            current_top = word["top"]
    lines.append(current_line)

    # Build output with paragraph gap detection
    output_parts: list[str] = []
    prev_line_bottom: Optional[float] = None

    for line in lines:
        # Sort words within line left-to-right
        line_sorted = sorted(line, key=lambda w: w["x0"])
        line_text = " ".join(w["text"] for w in line_sorted).strip()

        if not line_text:
            continue

        line_top = min(w["top"] for w in line)

        if prev_line_bottom is not None:
            gap = line_top - prev_line_bottom
            if gap >= paragraph_gap:
                output_parts.append("\n\n")
            else:
                output_parts.append("\n")

        output_parts.append(line_text)
        prev_line_bottom = max(w["bottom"] for w in line)

    return "".join(output_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Two-column layout detection and handling (ROOT-4)
# ─────────────────────────────────────────────────────────────────────────────

def detect_two_columns(words: list, page_width: float) -> Optional[float]:
    """
    Return the x-coordinate of the column split if the page looks two-column,
    else return None.

    Heuristic: build a histogram of word x0 positions. If there is a clear
    gap in the middle third of the page with very few word starts, it's likely
    a two-column layout.
    """
    if not words or page_width <= 0:
        return None

    mid_lo = page_width * 0.33
    mid_hi = page_width * 0.67
    mid_words = [w for w in words if mid_lo <= w["x0"] <= mid_hi]

    # If fewer than 5% of words start in the middle band → likely two columns
    if len(mid_words) / max(len(words), 1) < 0.05 and len(words) > 30:
        # Find the split point: x0 of the rightmost column's leftmost words
        right_col_starts = [
            w["x0"] for w in words if w["x0"] > mid_lo
        ]
        if right_col_starts:
            return min(right_col_starts)
    return None


def extract_page_text(page) -> str:
    """
    Extract text from a single pdfplumber Page object, handling:
    - Normal single-column layout
    - Two-column layouts (marginal-note style and true two-column)
    - Fallback to extract_text() if word extraction yields nothing
    """
    words = page.extract_words(
        x_tolerance=2,       # tighter than default 3 → don't merge across column gaps
        y_tolerance=2,       # tighter than default 3 → preserve distinct lines
        keep_blank_chars=False,
        use_text_flow=False,
    )

    if not words:
        # Fallback: try pdfplumber's built-in layout extraction
        fallback = page.extract_text(layout=False) or ""
        return fallback.strip()

    page_width = page.width

    # Check for two-column layout
    split_x = detect_two_columns(words, page_width)

    if split_x is not None:
        # Extract left column, then right column, concatenate
        left_words  = [w for w in words if w["x1"] <= split_x]
        right_words = [w for w in words if w["x0"] >= split_x]
        left_text   = words_to_structured_text(left_words)
        right_text  = words_to_structured_text(right_words)
        # Join with a clear separator so the cleaner can identify column boundaries
        parts = [p for p in [left_text, right_text] if p.strip()]
        return "\n\n".join(parts)
    else:
        return words_to_structured_text(words)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic paragraph injection for persistent single-line pages (ROOT-1 FIX-B)
# ─────────────────────────────────────────────────────────────────────────────

_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|i\.e|e\.g|"
    r"No|Sec|Art|Cl|cl|Ch|ss|sub|para|pt|vol|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"Ltd|Co|Corp|Inc|Pvt|Govt|Dept|Min|Div|"
    r"approx|est|ref|viz)\.",
    re.IGNORECASE,
)

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
    r"THE\s+SCHEDULE",
    r"Statement\s+of\s+Objects",
]

def inject_paragraph_breaks(text: str) -> str:
    """
    Last-resort heuristic: recover structure from text that is still a single
    blob after word-based extraction.  Applied only when newline count is ≤ 2.

    Rules applied in order:
    1. Legal provision keywords (BE it enacted, Provided that, etc.) → \\n\\n before
    2. Section numbers "N. Title" → \\n\\n before (when not an abbreviation)
    3. Sub-clauses "(1)", "(a)" → \\n before
    4. Sentence boundary ". Capital" → \\n (skipping known abbreviations)
    """
    # Rule 1: provision starters
    for starter in _PROVISION_STARTERS:
        text = re.sub(
            r"(?<!\n)(" + starter + r")",
            r"\n\n\1",
            text,
            flags=re.IGNORECASE,
        )

    # Rule 2: section numbers "42. Powers" — only when preceded by whitespace
    text = re.sub(
        r"(?<=\s)(\d{1,3})\.\s+([A-Z][a-z]{2,})",
        r"\n\n\1. \2",
        text,
    )

    # Rule 3: sub-clauses
    text = re.sub(r"\s+\((\d{1,3})\)\s+([A-Za-z])", r"\n(\1) \2", text)
    text = re.sub(r"\s+\(([a-z])\)\s+([A-Za-z])", r"\n(\1) \2", text)

    # Rule 4: sentence boundaries ". Capital" — skip abbreviations
    tokens = re.split(r"(\. )(?=[A-Z])", text)
    out: list[str] = []
    for i, tok in enumerate(tokens):
        if tok == ". " and i > 0:
            prev = "".join(out)
            last_m = re.search(r"(\b\w+)$", prev)
            is_abbrev    = last_m and _ABBREV_RE.search(last_m.group(1) + ".")
            is_digit     = last_m and re.match(r"^\d+$", last_m.group(1))
            if is_abbrev or is_digit:
                out.append(". ")
            else:
                out.append(".\n")
        else:
            out.append(tok)
    text = "".join(out)

    # Normalise blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def maybe_inject_breaks(text: str) -> str:
    """Apply inject_paragraph_breaks only when the page is still a blob."""
    newline_count = text.count("\n")
    word_count    = len(text.split())
    # Only inject if content is substantial but structure-free
    if word_count > 80 and newline_count <= 2:
        return inject_paragraph_breaks(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf(pdf_path: str) -> str:
    """Extract and reconstruct structured text from a PDF file."""
    full_text_parts: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                page_text = extract_page_text(page)
            except Exception as exc:
                log.warning("  Page %d extraction error: %s", page_num, exc)
                page_text = ""

            if not page_text.strip():
                continue

            # Apply heuristic break injection per page (ROOT-1 FIX-B)
            page_text = maybe_inject_breaks(page_text)
            full_text_parts.append(page_text)

    # ROOT-2: separate pages with paragraph break, not just '\n'
    return "\n\n".join(full_text_parts)


def collect_pdfs(root_folder: str) -> list[tuple[str, str]]:
    """
    Walk input folder and return (pdf_path, relative_output_path) pairs.
    ROOT-6: use relative subdirectory to avoid filename collisions.
    """
    pairs: list[tuple[str, str]] = []
    root = Path(root_folder)
    for pdf_path in root.rglob("*.pdf"):
        rel = pdf_path.relative_to(root)
        out_rel = rel.with_suffix(".txt")
        pairs.append((str(pdf_path), str(out_rel)))
    return pairs


def main() -> None:
    pdf_files = collect_pdfs(INPUT_FOLDER)
    if not pdf_files:
        log.error("No PDFs found in '%s'", INPUT_FOLDER)
        return

    scanned_log: list[str] = []

    for pdf_path, out_rel in tqdm(pdf_files, desc="Extracting"):
        out_path = Path(OUTPUT_FOLDER) / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)  # ROOT-6

        text = extract_pdf(pdf_path)
        word_count = len(text.split())

        # ROOT-3: detect and flag scanned / image-only PDFs
        if word_count < MIN_WORDS_THRESHOLD:
            sentinel = (
                f"[EXTRACTION_FAILED: {pdf_path}]\n"
                f"Only {word_count} words extracted — this PDF is likely a scanned "
                f"image without embedded text.\n"
                f"Re-extract using an OCR pipeline (e.g. pytesseract + pdf2image).\n"
            )
            out_path.write_text(sentinel, encoding="utf-8")
            scanned_log.append(pdf_path)
            log.warning("SCANNED (no text): %s", pdf_path)
            continue

        out_path.write_text(text, encoding="utf-8")

    print(f"\nExtraction complete. {len(pdf_files)} PDFs processed.")
    if scanned_log:
        print(f"\n⚠  {len(scanned_log)} PDFs need OCR (written as sentinel files):")
        for p in scanned_log:
            print(f"   {p}")
        print("\nRe-run these with an OCR extractor, e.g.:")
        print("   pip install pytesseract pdf2image")
        print("   Then use the ocr_extract.py companion script.")


if __name__ == "__main__":
    main()