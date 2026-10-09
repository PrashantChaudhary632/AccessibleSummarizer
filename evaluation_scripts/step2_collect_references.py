# evaluation_scripts/step2_collect_references.py
# ===============================================
# Verifies all human reference files exist.
# Run this after collecting written summaries from E1, E2, E3.

import os

REF_DIR = "evaluation/human_references"
DOCS    = ["NEP", "CrPC", "CGST", "NRHM", "JK"]
WRITERS = ["E1", "E2", "E3"]

print("Checking human reference files...\n")

missing = []
found   = []

for doc in DOCS:
    for writer in WRITERS:
        fname = f"{doc}_{writer}.txt"
        fpath = os.path.join(REF_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, encoding="utf-8") as f:
                words = len(f.read().split())
            found.append(fname)
            status = "OK" if words >= 50 else "WARNING: very short"
            print(f"  FOUND   {fname:<20} ({words} words) {status}")
        else:
            missing.append(fname)
            print(f"  MISSING {fname}")

print(f"\n  Found   : {len(found)}/15")
print(f"  Missing : {len(missing)}/15")

if missing:
    print("\n  Create these files in evaluation/human_references/")
    print("  Each file = one evaluator's written summary (150-200 words)")
else:
    print("\n  All reference files present! Run step3_evaluate_human.py next.")