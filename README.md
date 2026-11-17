# Cancer Trial Discovery for Hospital Research Teams

BSNS8030 Data Analytics · SDG 3 Good Health and Well-being · CRISP-DM

A Streamlit prototype that helps hospital clinical research teams find potentially relevant
cancer studies for clinical review. **Search supports discovery. It does not diagnose disease
or determine patient eligibility.**

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app: trial search, exploratory clusters, model evaluation, data and preparation, limitations |
| `pipeline.py` | Data preparation, TF-IDF / SVD / K-means modelling, evaluation and search (mirrors the group notebook) |
| `train_model.py` | Offline script: 42-configuration model selection, final fit, held-out evaluation; writes `artifacts/` |
| `artifacts/` | Saved results used by the app (model summary, 42 experiments, clusters, clinician review template) |
| `data/cancer_trials_10000.csv.gz` | 10,000-study ClinicalTrials.gov API v2 snapshot (gzip, read directly by pandas) |
| `requirements.txt` | Packages Streamlit Cloud installs |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
python train_model.py         # optional: re-run model selection and evaluation (about 20 seconds)
```

The app refits only the selected configuration at start-up (a few seconds), so no pickled model
files are needed and it works with any recent scikit-learn version.

## Deploy on Streamlit Community Cloud

1. Upload every file and folder above to a public GitHub repository (keep `app.py` at the top level).
2. On share.streamlit.io, choose **Create app → Deploy a public app from GitHub**, select the repository,
   branch `main` and main file `app.py`.

## Method (from the group notebook)

1. **Preparation:** valid unique NCT IDs, interventional studies only, `model_text` = cleaned conditions +
   interventions + title + summary combined **once** (no weighting), text ≥ 25 characters, recruiting flags.
2. **Splits:** 80/20 train/test (`random_state=42`), then 75/25 inner-train/validation inside training (60/20/20).
3. **Model selection:** TF-IDF (8,000 uni+bigrams) → TruncatedSVD + Normalizer (2–100 dimensions) → K-means
   (K = 3–12): 42 configurations scored on validation (50% top-10 coverage, 20% candidate reduction,
   15% SVD variance, 15% silhouette).
4. **Final model:** 100 dimensions, K = 6 (as presented), refitted on the 80% training split.
5. **Decision:** cluster-guided search (nearest 3 centroids) is adopted only if, on the untouched test set,
   it keeps ≥ 95% of the full-cosine top 10 while searching ≤ 70% of recruiting studies. It did not, so the
   app uses **full cosine ranking**; K-means stays as an exploratory view.

## Results on this data snapshot

| Measure | Value |
|---|---|
| Model-ready studies | 7,921 (train 6,336 · test 1,585) |
| TF-IDF variance retained (100D) | 23.76% |
| Held-out top-10 coverage (cluster-guided vs full cosine) | 79.17% (target ≥ 95%) |
| Held-out share of recruiting studies searched | 66.90% (target ≤ 70%) |
| Silhouette (train / test) | 0.051 / 0.054 |
| Adopted search mode | full cosine |

The presentation's figures (7,873 studies, 89.17% coverage, 70.04% searched) come from the notebook's own
live API download, which returned a slightly different snapshot. The method and the final decision are the same.
On this CSV the selection rule would narrowly prefer K = 5; set `USE_PRESENTED_CONFIG = False` in
`train_model.py` to use the rule's choice instead.

## Limitations

- Precision@10 not calculated: no independent clinician relevance labels (template in `artifacts/`).
- Historical snapshot: recheck recruitment status in the live registry.
- Ranked studies do not establish patient eligibility.
- Top-10 coverage is overlap with full-cosine ranking, not clinical accuracy.
