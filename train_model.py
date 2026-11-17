"""
Offline training and evaluation script (run once, then commit the artifacts/ folder).

    python train_model.py

Reproduces the group notebook end to end from data/cancer_trials_10000.csv.gz:
model selection (42 configurations), final fit on the 80% training split, held-out
evaluation, search-mode decision and the clinician review template.
The Streamlit app reads the small CSV/JSON results saved here and refits only the
selected configuration at start-up (a few seconds), so no pickled models are needed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline as P

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "cancer_trials_10000.csv.gz"
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

raw = pd.read_csv(DATA)
prepared, log = P.prepare(raw)
train_df, test_df, inner_df, val_df = P.split_data(prepared)
print("Prepared:", len(prepared), "| train", len(train_df), "test", len(test_df),
      "| inner", len(inner_df), "validation", len(val_df))

# The configuration reported in the group presentation (validation selected 100D, K=6 on the
# notebook's own API snapshot). Set USE_PRESENTED_CONFIG = False to let the selection rule
# choose again on this CSV snapshot.
USE_PRESENTED_CONFIG = True
PRESENTED = {"svd_dimensions": 100, "k": 6}

comp = P.run_model_selection(inner_df, val_df)
rule_best = P.select_configuration(comp)
comp.to_csv(OUT / "model_selection_all_experiments.csv", index=False)
if USE_PRESENTED_CONFIG:
    best = comp.loc[(comp["svd_dimensions"] == PRESENTED["svd_dimensions"]) & (comp["k"] == PRESENTED["k"])].iloc[0]
else:
    best = rule_best
D, K = int(best["svd_dimensions"]), int(best["k"])
print("Rule choice on this snapshot:", int(rule_best["svd_dimensions"]), "D, K =", int(rule_best["k"]))
print("Configuration used:", D, "dimensions, K =", K)

model = P.fit_final_model(train_df, D, K)
X_tr = model.vectorizer.transform(train_df["model_text"]); Z_tr = model.reducer.transform(X_tr)
X_te = model.vectorizer.transform(test_df["model_text"]); Z_te = model.reducer.transform(X_te)
lab_tr, lab_te = model.kmeans.predict(Z_tr), model.kmeans.predict(Z_te)
sil_tr = P.sampled_silhouette(Z_tr, lab_tr, 1000)
sil_te = P.sampled_silhouette(Z_te, lab_te, 1000)
cov_te, frac_te = P.retrieval_proxy_metrics(model.vectorizer, model.reducer, model.kmeans, X_te, Z_te, test_df)
mode = "cluster_guided" if (cov_te >= 0.95 and frac_te <= 0.70) else "full_cosine"

summary = {
    "svd_dimensions": D, "k": K,
    "explained_variance": model.explained_variance,
    "records": {"prepared": len(prepared), "train": len(train_df), "test": len(test_df),
                "inner_train": len(inner_df), "validation": len(val_df)},
    "preparation_log": log,
    "validation": {"top10_coverage": float(best["validation_direct_top10_coverage"]),
                   "candidate_fraction": float(best["validation_candidate_fraction"]),
                   "silhouette": float(best["silhouette_validation"]),
                   "selection_score": float(best["selection_score"])},
    "held_out_test": {"train_silhouette": sil_tr, "test_silhouette": sil_te,
                      "top10_coverage": cov_te, "candidate_fraction": frac_te},
    "thresholds": {"coverage_min": 0.95, "candidate_fraction_max": 0.70},
    "search_mode": mode,
    "configuration_source": "presented" if USE_PRESENTED_CONFIG else "selection_rule",
    "rule_choice_on_this_snapshot": {"svd_dimensions": int(rule_best["svd_dimensions"]), "k": int(rule_best["k"])},
    "tfidf_features": int(X_tr.shape[1]),
}
(OUT / "model_summary.json").write_text(json.dumps(summary, indent=2))

terms = P.cluster_top_terms(model)
pd.DataFrame({"cluster": list(terms), "top_terms": [", ".join(v) for v in terms.values()],
              "train_records": np.bincount(lab_tr, minlength=K),
              "test_records": np.bincount(lab_te, minlength=K)}).to_csv(OUT / "clusters.csv", index=False)

rows = []
for q in P.EVALUATION_QUERIES:
    full, guided, *_ = P.rank_trial_search(q, test_df, X_te, Z_te, model)
    for df in (full, guided):
        for _, r in df.iterrows():
            rows.append({"query": q, "method": r["Method"], "rank": r["Rank"], "NCT Number": r["NCT Number"],
                         "Study Title": r["Study Title"], "Study URL": r["Study URL"],
                         "clinician_relevance_0_or_1": ""})
pd.DataFrame(rows).to_csv(OUT / "clinician_relevance_review_template.csv", index=False)
print(json.dumps(summary["held_out_test"], indent=1), "\nSearch mode:", mode)
