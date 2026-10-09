# evaluation_scripts/step4_rouge_human_ref.py
# ============================================
# FIXED VERSION — uses your actual filenames
# Computes ROUGE scores using human-written reference summaries.
# Run after all 15 human reference files are saved in:
#   evaluation/human_references/
# Produces better ROUGE numbers for Table III in your paper.

import os
import json
from rouge_score import rouge_scorer

SUMMARY_DIR = "output/summaries"
REF_DIR     = "evaluation/human_references"
METRICS_DIR = "output/metrics"

os.makedirs(METRICS_DIR, exist_ok=True)

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
)

# Map: display name → (exact summary file stem, reference prefix)
# Summary file stem = filename inside output/summaries/ without _summary.txt
DOCUMENTS = {
    "NEP 2020" : ("NEP_Final_English_0_summary",
                  "NEP"),
    "CrPC"     : ("the_code_of_criminal_procedure,_1973_summary",
                  "CrPC"),
    "CGST Act" : ("CGST-Act-Updated-30092020_summary",
                  "CGST"),
    "NRHM"     : ("nrhm-framework-latest_summary",
                  "NRHM"),
    "JK Act"   : ("J&K Reorganisation Act, 2019_summary",
                  "JK"),
}

WRITERS = ["E1", "E2", "E3"]

print("\n" + "=" * 60)
print("  ROUGE WITH HUMAN REFERENCE SUMMARIES")
print("=" * 60)

all_results = []
skipped     = []

for doc_name, (sum_stem, ref_prefix) in DOCUMENTS.items():

    print(f"\n  Processing: {doc_name}")

    # ── Load system summary ───────────────────────────────────
    sum_path = os.path.join(SUMMARY_DIR, sum_stem + ".txt")

    if not os.path.exists(sum_path):
        print(f"    ERROR: Summary not found — {sum_path}")
        skipped.append(doc_name)
        continue

    with open(sum_path, encoding="utf-8", errors="replace") as f:
        system_summary = f.read()

    print(f"    Summary loaded  : {sum_path}")

    # ── Load human reference summaries ────────────────────────
    refs        = []
    refs_loaded = []

    for writer in WRITERS:
        ref_path = os.path.join(REF_DIR, f"{ref_prefix}_{writer}.txt")
        if os.path.exists(ref_path):
            with open(ref_path, encoding="utf-8", errors="replace") as f:
                content = f.read().strip()
            if content:
                refs.append(content)
                refs_loaded.append(writer)
            else:
                print(f"    WARNING: {ref_prefix}_{writer}.txt is empty")
        else:
            print(f"    WARNING: Missing — {ref_path}")

    if not refs:
        print(f"    SKIPPING — no reference files found for {doc_name}")
        print(f"    Create these files in evaluation/human_references/:")
        for w in WRITERS:
            print(f"      {ref_prefix}_{w}.txt")
        skipped.append(doc_name)
        continue

    print(f"    References used : {refs_loaded}")

    # ── Score against each reference and average ──────────────
    r1s, r2s, rls = [], [], []

    for ref in refs:
        s = scorer.score(ref, system_summary)
        r1s.append(s["rouge1"].fmeasure)
        r2s.append(s["rouge2"].fmeasure)
        rls.append(s["rougeL"].fmeasure)

    avg_r1 = sum(r1s) / len(r1s)
    avg_r2 = sum(r2s) / len(r2s)
    avg_rl = sum(rls) / len(rls)

    all_results.append({
        "document" : doc_name,
        "rouge1_f" : round(avg_r1, 4),
        "rouge2_f" : round(avg_r2, 4),
        "rougeL_f" : round(avg_rl, 4),
        "n_refs"   : len(refs),
    })

    print(f"    ROUGE-1 : {avg_r1:.4f}")
    print(f"    ROUGE-2 : {avg_r2:.4f}")
    print(f"    ROUGE-L : {avg_rl:.4f}")

# ── Overall average ───────────────────────────────────────────

print("\n" + "=" * 60)

if not all_results:
    print("  No results computed.")
    print("  Make sure human reference files exist in evaluation/human_references/")
    print("  File naming: NEP_E1.txt, NEP_E2.txt, NEP_E3.txt")
    print("               CrPC_E1.txt, CrPC_E2.txt, CrPC_E3.txt")
    print("               CGST_E1.txt, CGST_E2.txt, CGST_E3.txt")
    print("               NRHM_E1.txt, NRHM_E2.txt, NRHM_E3.txt")
    print("               JK_E1.txt,   JK_E2.txt,   JK_E3.txt")
else:
    n      = len(all_results)
    avg_r1 = sum(r["rouge1_f"] for r in all_results) / n
    avg_r2 = sum(r["rouge2_f"] for r in all_results) / n
    avg_rl = sum(r["rougeL_f"] for r in all_results) / n

    print("  OVERALL AVERAGE (paste into Table III)")
    print("=" * 60)
    print(f"  Documents scored : {n} / {len(DOCUMENTS)}")
    if skipped:
        print(f"  Skipped          : {skipped}")
    print(f"  ROUGE-1          : {avg_r1:.4f}")
    print(f"  ROUGE-2          : {avg_r2:.4f}")
    print(f"  ROUGE-L          : {avg_rl:.4f}")

    # ── Full Table III comparison ─────────────────────────────
    print("\n" + "=" * 65)
    print("  TABLE III — UPDATED ROUGE COMPARISON (copy to paper)")
    print("=" * 65)
    print(f"  {'Model':<44} {'R-1':>6} {'R-2':>6} {'R-L':>6}")
    print(f"  {'-'*62}")
    print(f"  {'Extractive Baseline (first 3 sentences)':<44}"
          f" {'0.1840':>6} {'0.1815':>6} {'0.1840':>6}")
    print(f"  {'Our System BART (vs first 1000w source)':<44}"
          f" {'0.2753':>6} {'0.1020':>6} {'0.1430':>6}")
    print(f"  {'Our System BART (vs human references)':<44}"
          f" {avg_r1:>6.4f} {avg_r2:>6.4f} {avg_rl:>6.4f}")
    print(f"  {'LexSum Stanford 2022 (published benchmark)':<44}"
          f" {'0.2700':>6} {'0.0800':>6} {'0.2200':>6}")
    print("=" * 65)

    # ── Per document breakdown ────────────────────────────────
    print("\n  PER DOCUMENT BREAKDOWN")
    print(f"  {'Document':<12} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'Refs':>5}")
    print(f"  {'-'*45}")
    for r in all_results:
        print(f"  {r['document']:<12} {r['rouge1_f']:>8.4f} "
              f"{r['rouge2_f']:>8.4f} {r['rougeL_f']:>8.4f} "
              f"{r['n_refs']:>5}")

    # ── Save JSON ─────────────────────────────────────────────
    output = {
        "per_document" : all_results,
        "average"      : {
            "rouge1" : round(avg_r1, 4),
            "rouge2" : round(avg_r2, 4),
            "rougeL" : round(avg_rl, 4),
        },
        "note"    : "Averaged across E1, E2, E3 written references",
        "skipped" : skipped,
    }

    json_path = os.path.join(METRICS_DIR, "rouge_human_ref.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved to {json_path}")
    print("  Use the 'vs human references' row in Table III of your paper.")
    print("=" * 65 + "\n")