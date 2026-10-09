# evaluation_scripts/step1_prepare_excerpts.py
# =============================================
# FIXED VERSION — uses your actual filenames
# Extracts first 400 words from each evaluation document
# and copies the accessible summary alongside it.
# Run this FIRST before giving anything to evaluators.

import os
import shutil

# ── EXACT filenames from your dataset/ready_text folder ──────
DOCUMENTS = {
    "NEP_2020"  : "NEP_Final_English_0",
    "CrPC"      : "the_code_of_criminal_procedure,_1973",
    "CGST_Act"  : "CGST-Act-Updated-30092020",
    "NRHM"      : "nrhm-framework-latest",
    "JK_Act"    : "J&K Reorganisation Act, 2019",
}

SOURCE_DIR     = "dataset/ready_text"
ACCESSIBLE_DIR = "output/accessible"
EXCERPT_DIR    = "evaluation/source_excerpts"

os.makedirs(EXCERPT_DIR, exist_ok=True)

print("Preparing evaluation packets...\n")
print("=" * 60)

found_count   = 0
missing_count = 0

for display_name, file_stem in DOCUMENTS.items():

    print(f"\nProcessing: {display_name}")
    print(f"  Looking for: {file_stem}.txt")

    # ── Source excerpt ────────────────────────────────────────
    src_path = os.path.join(SOURCE_DIR, file_stem + ".txt")

    if not os.path.exists(src_path):
        print(f"  ERROR: Source file not found!")
        print(f"         Expected: {src_path}")
        missing_count += 1
        continue

    with open(src_path, encoding="utf-8", errors="replace") as f:
        words = f.read().split()

    excerpt = " ".join(words[:400])

    out_src = os.path.join(EXCERPT_DIR, f"{display_name}_SOURCE.txt")
    with open(out_src, "w", encoding="utf-8") as f:
        f.write(f"DOCUMENT: {display_name.replace('_', ' ')}\n")
        f.write("=" * 60 + "\n")
        f.write("(First 400 words shown for evaluation)\n")
        f.write("=" * 60 + "\n\n")
        f.write(excerpt)

    print(f"  Source excerpt saved  : {out_src}  ({len(words)} total words)")

    # ── Accessible summary ────────────────────────────────────
    acc_path = os.path.join(ACCESSIBLE_DIR, file_stem + "_accessible.txt")

    if not os.path.exists(acc_path):
        print(f"  WARNING: Accessible summary not found!")
        print(f"           Expected: {acc_path}")
        print(f"           Run accessibility_layer.py first, then retry.")

        # Try to find any accessible file that contains the stem partially
        candidates = [
            f for f in os.listdir(ACCESSIBLE_DIR)
            if display_name.lower().replace("_", "") in f.lower().replace("_", "")
        ] if os.path.exists(ACCESSIBLE_DIR) else []

        if candidates:
            print(f"  HINT: Did you mean one of these?")
            for c in candidates:
                print(f"         {c}")
        continue

    out_acc = os.path.join(EXCERPT_DIR, f"{display_name}_SUMMARY.txt")
    shutil.copy(acc_path, out_acc)
    print(f"  Accessible summary    : {out_acc}")
    found_count += 1

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  DONE")
print("=" * 60)
print(f"  Successfully prepared : {found_count} / {len(DOCUMENTS)} documents")

if missing_count > 0:
    print(f"  Missing              : {missing_count} documents (see errors above)")

print(f"\n  Files saved to: {EXCERPT_DIR}/")
print()
print("  What to give each evaluator:")
print("  ─────────────────────────────────────────────")
for display_name in DOCUMENTS:
    src  = os.path.join(EXCERPT_DIR, f"{display_name}_SOURCE.txt")
    summ = os.path.join(EXCERPT_DIR, f"{display_name}_SUMMARY.txt")
    src_ok  = "OK" if os.path.exists(src)  else "MISSING"
    summ_ok = "OK" if os.path.exists(summ) else "MISSING"
    print(f"  {display_name:<12}  SOURCE [{src_ok}]   SUMMARY [{summ_ok}]")

print()
print("  Give evaluators:")
print("  *_SOURCE.txt  — they read this first (original excerpt)")
print("  *_SUMMARY.txt — they rate this (system generated)")
print("=" * 60)