"""
accessibility_layer.py
======================
Phase 6 Step 5 — Accessibility Layer (The Project Novelty)

Converts each BART summary into simplified bullet-point key points
that are understandable by non-expert citizens.

Pipeline:
  Summary text
       ↓
  Sentence scoring (importance + simplicity)
       ↓
  Key point extraction (top N sentences)
       ↓
  Simplification (remove legal jargon)
       ↓
  Bullet point output

Also computes:
  - Flesch-Kincaid readability score for source vs summary vs accessible version
  - Compression ratio

Usage:
    python accessibility_layer.py

Input  : output/summaries/*.txt
Output : output/accessible/*.txt
         output/metrics/readability_scores.json
"""

import os
import re
import json
from pathlib import Path
from collections import Counter

SUMMARY_DIR    = "output/summaries"
ACCESSIBLE_DIR = "output/accessible"
METRICS_DIR    = "output/metrics"
SOURCE_DIR     = "dataset/ready_text"

os.makedirs(ACCESSIBLE_DIR, exist_ok=True)
os.makedirs(METRICS_DIR,    exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Readability: Flesch-Kincaid Reading Ease
# FK = 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
# Higher = easier. 60-70 = standard. 30-50 = difficult. <30 = very difficult.
# ─────────────────────────────────────────────────────────────────────────────

def count_syllables(word: str) -> int:
    word = word.lower().rstrip(".,;:!?\"'")
    if len(word) <= 3:
        return 1
    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e"):
        count -= 1
    return max(1, count)


def flesch_kincaid(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words     = re.findall(r"\b[a-zA-Z]+\b", text)
    if not sentences or not words:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    n_sent    = len(sentences)
    n_words   = len(words)
    fk = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (syllables / n_words)
    return round(fk, 2)


def fk_label(score: float) -> str:
    if score >= 70:  return "Easy (general public)"
    if score >= 60:  return "Standard (10th grade)"
    if score >= 50:  return "Fairly difficult"
    if score >= 30:  return "Difficult (legal/academic)"
    return "Very difficult"


# ─────────────────────────────────────────────────────────────────────────────
# Jargon simplification dictionary
# Maps common legal/bureaucratic terms to plain English
# ─────────────────────────────────────────────────────────────────────────────

_SIMPLIFY = {
    "shall":                    "will",
    "hereinafter":              "from now on",
    "pursuant to":              "according to",
    "notwithstanding":          "despite",
    "in accordance with":       "following",
    "is hereby":                "is",
    "the aforementioned":       "the above",
    "inter alia":               "among other things",
    "prima facie":              "at first glance",
    "cognizable":               "reportable",
    "remuneration":             "pay",
    "endeavour":                "try",
    "thereof":                  "of it",
    "wherein":                  "where",
    "therein":                  "in it",
    "hereunder":                "below",
    "forthwith":                "immediately",
    "adjudication":             "decision",
    "promulgate":               "announce",
    "Consolidated Fund":        "government treasury",
    "Comptroller":              "financial auditor",
    "superintendence":          "oversight",
}


def simplify_sentence(sent: str) -> str:
    result = sent
    for legal, plain in _SIMPLIFY.items():
        result = re.sub(
            r"\b" + re.escape(legal) + r"\b",
            plain,
            result,
            flags=re.IGNORECASE,
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Key point extraction
# Score sentences by:
#   1. TF of content words (importance)
#   2. Sentence length penalty (prefer 10-25 words — readable bullets)
#   3. Position bonus (first sentences of summary = more important)
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has",
    "had","will","would","could","should","may","might","shall","this","that",
    "these","those","it","its","as","not","no","any","all","which","who",
    "their","they","he","she","we","also","such","into","than","there",
}

MAX_BULLETS = 7
MIN_BULLETS = 3


def extract_key_points(summary_text: str, n: int = MAX_BULLETS) -> list:
    """Extract top N key point sentences from a summary."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary_text)
                 if len(s.split()) >= 6]

    if not sentences:
        return []

    # TF scoring
    all_words = []
    for s in sentences:
        tokens = re.findall(r"\b[a-z]+\b", s.lower())
        all_words.extend(t for t in tokens if t not in _STOPWORDS)
    freq = Counter(all_words)

    scored = []
    for i, sent in enumerate(sentences):
        tokens = re.findall(r"\b[a-z]+\b", sent.lower())
        content = [t for t in tokens if t not in _STOPWORDS]
        if not content:
            continue
        tf_score = sum(freq[t] for t in content) / len(content)
        # Prefer sentences of 10-25 words
        wc = len(sent.split())
        len_bonus = 1.3 if 10 <= wc <= 25 else (0.8 if wc > 40 else 1.0)
        # Position bonus (earlier = more important in summary)
        pos_bonus = 1.2 if i < 3 else 1.0
        scored.append((tf_score * len_bonus * pos_bonus, i, sent))

    # Pick top N, restore original order
    top_idx = sorted(
        [idx for _, idx, _ in sorted(scored, reverse=True)[:n]]
    )
    return [sentences[i] for i in top_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Main accessibility converter
# ─────────────────────────────────────────────────────────────────────────────

def get_summary_body(path: str) -> str:
    text  = open(path, encoding="utf-8", errors="replace").read()
    lines = text.split("\n")
    body_lines, in_body = [], False
    for line in lines:
        if line.startswith("====="):
            in_body = not in_body
            continue
        if in_body:
            body_lines.append(line)
    body = "\n".join(body_lines).strip()
    body = re.sub(r"\nWords:.*$", "", body).strip()
    return body


def make_accessible(summary_text: str, doc_title: str) -> str:
    """
    Convert a BART summary into an accessible bullet-point document.
    This is the novelty step: structured accessibility output.
    """
    n_bullets = max(MIN_BULLETS, min(MAX_BULLETS, len(summary_text.split()) // 40))
    key_points = extract_key_points(summary_text, n=n_bullets)

    if not key_points:
        key_points = [s.strip() for s in summary_text.split(".")
                      if len(s.split()) >= 6][:MIN_BULLETS]

    # Simplify each key point
    simplified = [simplify_sentence(kp) for kp in key_points]

    # Build output
    lines = [
        f"ACCESSIBLE SUMMARY: {doc_title}",
        "=" * 60,
        "",
        "KEY POINTS",
        "-" * 30,
    ]
    for i, point in enumerate(simplified, 1):
        # Ensure sentence ends with period
        point = point.strip()
        if point and not point.endswith((".", "!", "?")):
            point += "."
        # Capitalise first letter
        if point:
            point = point[0].upper() + point[1:]
        lines.append(f"  {i}. {point}")

    lines += [
        "",
        "PLAIN LANGUAGE SUMMARY",
        "-" * 30,
        summary_text[:500].strip() + ("..." if len(summary_text) > 500 else ""),
        "",
        "=" * 60,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

def main():
    summary_files = sorted(
        f for f in os.listdir(SUMMARY_DIR) if f.endswith("_summary.txt")
    )

    if not summary_files:
        print(f"No summary files found in {SUMMARY_DIR}/")
        return

    print(f"Processing {len(summary_files)} summaries...")

    readability_records = []

    for fname in summary_files:
        source_fname = fname.replace("_summary.txt", ".txt")
        summary_path = os.path.join(SUMMARY_DIR, fname)
        source_path  = os.path.join(SOURCE_DIR, source_fname)

        summary_body = get_summary_body(summary_path)
        if not summary_body.strip():
            continue

        doc_title = Path(fname).stem.replace("_summary", "").replace("_", " ").replace("-", " ").title()

        # Build accessible version
        accessible = make_accessible(summary_body, doc_title)

        # Save
        out_path = os.path.join(ACCESSIBLE_DIR, fname.replace("_summary.txt", "_accessible.txt"))
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(accessible)

        # Readability scores
        fk_summary = flesch_kincaid(summary_body)

        # Extract key points text for readability comparison
        key_points = extract_key_points(summary_body)
        simplified_pts = [simplify_sentence(kp) for kp in key_points]
        accessible_text = " ".join(simplified_pts)
        fk_accessible = flesch_kincaid(accessible_text)

        # Source readability (first 500 words)
        fk_source = 0.0
        if os.path.exists(source_path):
            src = open(source_path, encoding="utf-8", errors="replace").read()
            fk_source = flesch_kincaid(" ".join(src.split()[:500]))

        record = {
            "file"            : source_fname,
            "fk_source"       : fk_source,
            "fk_summary"      : fk_summary,
            "fk_accessible"   : fk_accessible,
            "fk_label_source"     : fk_label(fk_source),
            "fk_label_summary"    : fk_label(fk_summary),
            "fk_label_accessible" : fk_label(fk_accessible),
            "improvement_summary_vs_source"    : round(fk_summary - fk_source, 2),
            "improvement_accessible_vs_source" : round(fk_accessible - fk_source, 2),
            "summary_words"   : len(summary_body.split()),
            "accessible_bullets" : len(key_points),
        }
        readability_records.append(record)

    if not readability_records:
        print("No records produced.")
        return

    # Aggregate
    n = len(readability_records)
    avg_src  = sum(r["fk_source"]       for r in readability_records) / n
    avg_sum  = sum(r["fk_summary"]      for r in readability_records) / n
    avg_acc  = sum(r["fk_accessible"]   for r in readability_records) / n
    avg_imp  = sum(r["improvement_accessible_vs_source"] for r in readability_records) / n

    # Save JSON
    output = {
        "aggregate": {
            "avg_fk_source"        : round(avg_src, 2),
            "avg_fk_summary"       : round(avg_sum, 2),
            "avg_fk_accessible"    : round(avg_acc, 2),
            "avg_improvement"      : round(avg_imp, 2),
            "label_source"         : fk_label(avg_src),
            "label_summary"        : fk_label(avg_sum),
            "label_accessible"     : fk_label(avg_acc),
        },
        "per_file": readability_records,
    }
    json_path = os.path.join(METRICS_DIR, "readability_scores.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Print report
    print(f"\n{'='*60}")
    print(f"  ACCESSIBILITY LAYER COMPLETE")
    print(f"{'='*60}")
    print(f"  Files processed        : {n}")
    print(f"  Accessible summaries  -> {ACCESSIBLE_DIR}/")
    print(f"\n  FLESCH-KINCAID READABILITY SCORES (higher = easier)")
    print(f"  Source documents  : {avg_src:.1f}  ({fk_label(avg_src)})")
    print(f"  BART summaries    : {avg_sum:.1f}  ({fk_label(avg_sum)})")
    print(f"  Accessible output : {avg_acc:.1f}  ({fk_label(avg_acc)})")
    print(f"\n  Avg improvement (accessible vs source): +{avg_imp:.1f} points")
    print(f"  Readability scores -> {json_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()