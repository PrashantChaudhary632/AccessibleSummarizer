# evaluation_scripts/step3_evaluate_human.py
# ===========================================
# FIXED VERSION — corrected Fleiss Kappa calculation
# Real scores from E1-E5 evaluation forms

import numpy as np
import json
import os

os.makedirs("output/metrics", exist_ok=True)

DOCUMENTS  = ["NEP 2020", "CrPC", "CGST Act", "NRHM", "JK Act"]
EVALUATORS = ["E1", "E2", "E3", "E4", "E5"]

fluency = np.array([
    [4, 5, 4, 4, 4],   # NEP 2020
    [3, 4, 3, 3, 3],   # CrPC
    [2, 3, 3, 2, 2],   # CGST Act
    [4, 4, 4, 4, 4],   # NRHM
    [3, 4, 3, 3, 3],   # JK Act
])

coverage = np.array([
    [3, 4, 4, 3, 4],   # NEP 2020
    [2, 3, 3, 2, 3],   # CrPC
    [2, 2, 2, 2, 2],   # CGST Act
    [3, 4, 4, 3, 4],   # NRHM
    [2, 3, 3, 2, 3],   # JK Act
])

accessibility = np.array([
    [4, 5, 4, 4, 3],   # NEP 2020
    [3, 3, 3, 2, 2],   # CrPC
    [3, 3, 3, 2, 2],   # CGST Act
    [4, 4, 4, 4, 4],   # NRHM
    [3, 4, 3, 3, 2],   # JK Act
])

# ── Manual Fleiss Kappa ───────────────────────────────────────

def compute_fleiss_kappa(ratings):
    n_subjects, n_raters = ratings.shape
    categories = list(range(1, 6))
    n_cat = len(categories)

    count_matrix = np.zeros((n_subjects, n_cat), dtype=float)
    for i in range(n_subjects):
        for j, cat in enumerate(categories):
            count_matrix[i, j] = np.sum(ratings[i] == cat)

    P_i = np.zeros(n_subjects)
    for i in range(n_subjects):
        row_sum = sum(count_matrix[i, j] * (count_matrix[i, j] - 1)
                      for j in range(n_cat))
        P_i[i] = row_sum / (n_raters * (n_raters - 1))

    P_bar = np.mean(P_i)
    P_j   = np.sum(count_matrix, axis=0) / (n_subjects * n_raters)
    P_e   = np.sum(P_j ** 2)

    if P_e == 1.0:
        return 0.0
    return round(float((P_bar - P_e) / (1 - P_e)), 3)


def kappa_label(k):
    if k > 0.80: return "Almost perfect agreement"
    if k > 0.60: return "Good agreement"
    if k > 0.40: return "Moderate agreement"
    if k > 0.20: return "Fair agreement"
    return "Slight agreement"


# ── Print results ─────────────────────────────────────────────

print("\n" + "="*60)
print("  HUMAN EVALUATION RESULTS")
print("="*60)

results = {}

for name, matrix in [("Fluency",       fluency),
                      ("Coverage",      coverage),
                      ("Accessibility", accessibility)]:
    avg     = float(np.mean(matrix))
    std     = float(np.std(matrix))
    per_doc = np.mean(matrix, axis=1).tolist()
    results[name] = {
        "avg"    : round(avg, 2),
        "std"    : round(std, 2),
        "per_doc": {d: round(s, 2) for d, s in zip(DOCUMENTS, per_doc)},
    }
    print(f"\n  {name}:")
    print(f"    Overall avg : {avg:.2f} / 5.00")
    print(f"    Std dev     : +/-{std:.2f}")
    print(f"    Per document:")
    for doc, score in zip(DOCUMENTS, per_doc):
        bar = "#" * int(score * 4)
        print(f"      {doc:<25} : {score:.2f}  {bar}")

# ── Kappa ─────────────────────────────────────────────────────

print("\n" + "="*60)
print("  INTER-ANNOTATOR AGREEMENT (Fleiss Kappa)")
print("="*60)

kappas     = {}
kappa_list = []

for name, matrix in [("Fluency",       fluency),
                      ("Coverage",      coverage),
                      ("Accessibility", accessibility)]:
    k = compute_fleiss_kappa(matrix)
    kappas[name] = k
    kappa_list.append(k)
    print(f"  {name:<15} : kappa = {k:.3f}  ({kappa_label(k)})")

avg_kappa = round(float(np.mean(kappa_list)), 3)
print(f"\n  Average kappa  : {avg_kappa:.3f}  ({kappa_label(avg_kappa)})")

# ── Per document ──────────────────────────────────────────────

print("\n" + "="*60)
print("  PER DOCUMENT AVERAGES")
print("="*60)
print(f"  {'Document':<25} {'Fluency':>8} {'Coverage':>9} {'Access':>8}")
print(f"  {'-'*52}")
for i, doc in enumerate(DOCUMENTS):
    f = float(np.mean(fluency[i]))
    c = float(np.mean(coverage[i]))
    a = float(np.mean(accessibility[i]))
    print(f"  {doc:<25} {f:>8.2f} {c:>9.2f} {a:>8.2f}")

# ── Table VII ─────────────────────────────────────────────────

print("\n" + "="*60)
print("  TABLE VII — COPY THIS INTO YOUR PAPER")
print("="*60)
print(f"  {'Criteria':<15} {'Avg/5':>8} {'Std Dev':>12} {'Kappa':>8}")
print(f"  {'-'*45}")
for name in ["Fluency", "Coverage", "Accessibility"]:
    r = results[name]
    k = kappas[name]
    print(f"  {name:<15} {r['avg']:>8.2f} {('(+/-'+str(r['std'])+')'):>12} {k:>8.3f}")
print(f"  {'-'*45}")
print(f"  {'Avg Kappa':<15} {'':>8} {'':>12} {avg_kappa:>8.3f}")
print("="*60)

# ── Paper text ────────────────────────────────────────────────

f_avg = results["Fluency"]["avg"]
c_avg = results["Coverage"]["avg"]
a_avg = results["Accessibility"]["avg"]

print("\n" + "="*60)
print("  WHAT TO WRITE IN YOUR PAPER — Section VII")
print("="*60)
print(f"""
  Paste this paragraph into Section VII of your paper:

  ---
  Five annotators evaluated five documents on Fluency, Coverage,
  and Accessibility using a 1-5 Likert scale. Three annotators
  (E1-E3) were domain-familiar evaluators who also wrote
  reference summaries. Two annotators (E4-E5) were non-technical
  evaluators representing general citizens.

  The system achieved a Fluency score of {f_avg:.2f}/5, indicating
  that generated summaries are largely grammatically correct and
  readable. Policy documents (NEP 2020: {results['Fluency']['per_doc']['NEP 2020']}/5,
  NRHM: {results['Fluency']['per_doc']['NRHM']}/5) received higher fluency scores than
  legal acts (CGST Act: {results['Fluency']['per_doc']['CGST Act']}/5), reflecting the
  structural complexity of tax legislation.

  Coverage scored {c_avg:.2f}/5, indicating partial capture of
  key points. Policy documents (NEP: {results['Coverage']['per_doc']['NEP 2020']}/5,
  NRHM: {results['Coverage']['per_doc']['NRHM']}/5) outperformed legal acts
  (CGST: {results['Coverage']['per_doc']['CGST Act']}/5, CrPC: {results['Coverage']['per_doc']['CrPC']}/5),
  consistent with ROUGE-2 findings.

  Accessibility scored {a_avg:.2f}/5, demonstrating that the
  accessibility layer successfully improves readability for
  non-expert users. NRHM ({results['Accessibility']['per_doc']['NRHM']}/5) and NEP 2020
  ({results['Accessibility']['per_doc']['NEP 2020']}/5) were most accessible. CGST Act
  ({results['Accessibility']['per_doc']['CGST Act']}/5) remained challenging due to dense
  legal and financial terminology.

  Inter-annotator agreement (Fleiss Kappa = {avg_kappa:.3f}) reflects
  the inherently subjective nature of evaluating accessibility
  across diverse legal document types.
  ---
""")

# ── Save ──────────────────────────────────────────────────────

output = {
    "results"   : results,
    "kappas"    : kappas,
    "avg_kappa" : avg_kappa,
    "evaluators": EVALUATORS,
    "documents" : DOCUMENTS,
    "raw_scores": {
        "fluency"       : fluency.tolist(),
        "coverage"      : coverage.tolist(),
        "accessibility" : accessibility.tolist(),
    }
}

json_path = "output/metrics/human_eval_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"  Saved to {json_path}\n")