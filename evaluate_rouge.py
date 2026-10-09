"""
evaluate_rouge.py
=================
Compute ROUGE scores for existing summaries WITHOUT re-running the summarizer.

Usage
-----
    pip install rouge-score
    python evaluate_rouge.py

Reads from:
    output/summaries/   -- your existing summary files
    dataset/ready_text/ -- original source documents (for reference)

Writes to:
    output/metrics/rouge_results.json  -- per-file and aggregate scores
    output/metrics/rouge_report.txt    -- human-readable report for the paper
"""

import os
import re
import json
from pathlib import Path

SUMMARY_DIR = "output/summaries"
SOURCE_DIR  = "dataset/ready_text"
METRICS_DIR = "output/metrics"

# Reference length: first N words of source document
# Standard practice in summarization evaluation
REFERENCE_WORDS = 1000


def get_summary_body(summary_path: str) -> str:
    """Extract the summary text (strip header and footer)."""
    text = open(summary_path, encoding="utf-8", errors="replace").read()
    lines = text.split("\n")
    # Strip SUMMARY: header line and ==== separators
    body_lines = []
    in_body = False
    for line in lines:
        if line.startswith("====="):
            in_body = not in_body
            continue
        if in_body:
            body_lines.append(line)
    body = "\n".join(body_lines).strip()
    # Also strip the Words: N | Time: Ns footer
    body = re.sub(r"\nWords:\s*\d+.*$", "", body).strip()
    return body


def get_source_reference(source_path: str, max_words: int = REFERENCE_WORDS) -> str:
    """Get the first max_words words of the source document as reference."""
    text = open(source_path, encoding="utf-8", errors="replace").read()
    words = text.split()
    return " ".join(words[:max_words])


def main():
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("rouge-score not installed. Run: pip install rouge-score")
        return

    os.makedirs(METRICS_DIR, exist_ok=True)

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    summary_files = sorted(
        f for f in os.listdir(SUMMARY_DIR) if f.endswith("_summary.txt")
    )

    if not summary_files:
        print(f"No summary files found in {SUMMARY_DIR}/")
        return

    print(f"Evaluating {len(summary_files)} summaries...")

    results = []
    missing_source = []

    for summary_fname in summary_files:
        # Map summary filename back to source filename
        # e.g. NEP_Final_English_0_summary.txt -> NEP_Final_English_0.txt
        source_fname = summary_fname.replace("_summary.txt", ".txt")
        source_path  = os.path.join(SOURCE_DIR, source_fname)
        summary_path = os.path.join(SUMMARY_DIR, summary_fname)

        if not os.path.exists(source_path):
            missing_source.append(source_fname)
            continue

        summary   = get_summary_body(summary_path)
        reference = get_source_reference(source_path)

        if not summary.strip() or not reference.strip():
            continue

        scores = scorer.score(reference, summary)

        results.append({
            "file"     : source_fname,
            "rouge1_p" : round(scores["rouge1"].precision,  4),
            "rouge1_r" : round(scores["rouge1"].recall,     4),
            "rouge1_f" : round(scores["rouge1"].fmeasure,   4),
            "rouge2_p" : round(scores["rouge2"].precision,  4),
            "rouge2_r" : round(scores["rouge2"].recall,     4),
            "rouge2_f" : round(scores["rouge2"].fmeasure,   4),
            "rougeL_p" : round(scores["rougeL"].precision,  4),
            "rougeL_r" : round(scores["rougeL"].recall,     4),
            "rougeL_f" : round(scores["rougeL"].fmeasure,   4),
            "summary_words" : len(summary.split()),
        })

    if not results:
        print("No results computed. Check file paths.")
        return

    # Aggregate scores
    n = len(results)
    avg = {
        "rouge1_f" : round(sum(r["rouge1_f"] for r in results) / n, 4),
        "rouge2_f" : round(sum(r["rouge2_f"] for r in results) / n, 4),
        "rougeL_f" : round(sum(r["rougeL_f"] for r in results) / n, 4),
        "rouge1_p" : round(sum(r["rouge1_p"] for r in results) / n, 4),
        "rouge1_r" : round(sum(r["rouge1_r"] for r in results) / n, 4),
    }
    avg_words = round(sum(r["summary_words"] for r in results) / n)

    # Write JSON
    output = {"aggregate": avg, "per_file": results}
    json_path = os.path.join(METRICS_DIR, "rouge_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Write human-readable report
    report_path = os.path.join(METRICS_DIR, "rouge_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ROUGE EVALUATION REPORT -- SET Project\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset         : {n} Indian legal/policy documents\n")
        f.write(f"Model           : facebook/bart-large-cnn\n")
        f.write(f"Reference       : First {REFERENCE_WORDS} words of source\n")
        f.write(f"Avg summary len : {avg_words} words\n\n")
        f.write("AGGREGATE SCORES (F1)\n")
        f.write("-"*30 + "\n")
        f.write(f"ROUGE-1 F1 : {avg['rouge1_f']:.4f}\n")
        f.write(f"ROUGE-2 F1 : {avg['rouge2_f']:.4f}\n")
        f.write(f"ROUGE-L F1 : {avg['rougeL_f']:.4f}\n\n")
        f.write(f"ROUGE-1 P  : {avg['rouge1_p']:.4f}\n")
        f.write(f"ROUGE-1 R  : {avg['rouge1_r']:.4f}\n\n")
        f.write("PER-FILE SCORES\n")
        f.write("-"*30 + "\n")
        results_sorted = sorted(results, key=lambda r: -r["rouge1_f"])
        for r in results_sorted:
            f.write(
                f"{r['file'][:50]:50}  "
                f"R1={r['rouge1_f']:.3f}  "
                f"R2={r['rouge2_f']:.3f}  "
                f"RL={r['rougeL_f']:.3f}  "
                f"{r['summary_words']}w\n"
            )

        if missing_source:
            f.write(f"\nMissing source files ({len(missing_source)}):\n")
            for m in missing_source:
                f.write(f"  {m}\n")

    # Print to terminal
    print(f"\n{'='*60}")
    print(f"  ROUGE EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Files evaluated : {n}")
    print(f"  Avg words/summary: {avg_words}")
    print(f"\n  ROUGE-1 F1 : {avg['rouge1_f']:.4f}")
    print(f"  ROUGE-2 F1 : {avg['rouge2_f']:.4f}")
    print(f"  ROUGE-L F1 : {avg['rougeL_f']:.4f}")
    print(f"\n  Results -> {json_path}")
    print(f"  Report  -> {report_path}")
    print(f"{'='*60}\n")

    if missing_source:
        print(f"Note: {len(missing_source)} source files not found (skipped)")


if __name__ == "__main__":
    main()