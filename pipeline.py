"""
Shared modelling pipeline for the hospital-facing Cancer Trial Discovery app.

Mirrors the group notebook "Cancer_Trials_Model_Text_Only_Explained.ipynb":
  Stage 3  Data preparation  -> prepare()
  Stage 4  Modelling         -> fit_final_model(), run_model_selection()
  Stage 5  Evaluation        -> retrieval_proxy_metrics(), rank_trial_search()

Key design choices (as in the notebook and presentation):
  * model_text = conditions + interventions + title + summary, cleaned, ONCE (no weighting)
  * 80/20 train/test split, then 75/25 inner-train/validation inside training (60/20/20)
  * TF-IDF (8,000 uni+bigram features) -> TruncatedSVD + Normalizer -> K-means
  * Search: query -> same TF-IDF -> cosine ranking of RECRUITING studies (full cosine).
    Cluster-guided search (nearest 3 centroids) is kept for comparison only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 8000
SVD_DIMENSIONS = [2, 5, 10, 25, 50, 100]
K_VALUES = [3, 4, 5, 6, 8, 10, 12]
TOP_N = 10
N_NEAREST_CLUSTERS = 3

EVALUATION_QUERIES = [
    "lung cancer immunotherapy", "breast cancer chemotherapy", "prostate cancer radiation therapy",
    "colorectal cancer targeted therapy", "ovarian cancer chemotherapy", "pancreatic cancer gemcitabine",
    "melanoma immune checkpoint inhibitor", "glioblastoma temozolomide",
    "lymphoma monoclonal antibody", "leukemia stem cell transplantation",
    "cervical cancer radiotherapy", "liver cancer ablation",
]

REQUIRED = ["NCT Number", "Study Title", "Study Status", "Brief Summary", "Conditions",
            "Interventions", "Study Type"]
OPTIONAL = ["Study URL", "Sponsor", "Collaborators", "Sex", "Age", "Phases", "Enrollment",
            "Countries", "Eligibility Criteria", "Minimum Age", "Maximum Age"]


# ----------------------------------------------------------------------------------------
# Stage 3 — Data preparation (notebook cells 25–43)
# ----------------------------------------------------------------------------------------
def _clean_text(series: pd.Series) -> pd.Series:
    """Lower-case and standardise tumour/tumours -> tumor (notebook cells 33–36)."""
    return series.fillna("").astype(str).str.lower().str.replace(r"\btumours?\b", "tumor", regex=True)


def tidy_phase(value) -> str:
    v = str(value or "").upper()
    if "EARLY_PHASE1" in v:
        return "Early Phase 1"
    nums = re.findall(r"PHASE(\d)", v)
    return "Phase " + "/".join(sorted(set(nums))) if nums else "Not recorded"


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    log = [("Raw records", len(raw))]
    df = raw.copy()
    for col in OPTIONAL:
        if col not in df.columns:
            df[col] = np.nan
    eligibility_display = df["Eligibility Criteria"].copy()   # keep line breaks for on-screen reading

    # Normalise whitespace in every text field
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x)

    # Valid, unique NCT identifiers
    df["NCT Number"] = df["NCT Number"].astype("string").str.upper().str.strip()
    df = df.loc[df["NCT Number"].str.fullmatch(r"NCT\d{8}", na=False)].copy()
    df = df.drop_duplicates(subset=["NCT Number"], keep="first").copy()
    log.append(("Valid NCT ID, deduplicated", len(df)))

    # Standardise categories, keep interventional studies
    df["Study Type"] = df["Study Type"].astype("string").str.upper().str.replace(" ", "_", regex=False)
    df["Study Status"] = df["Study Status"].astype("string").str.upper().str.replace(" ", "_", regex=False)
    df = df.loc[df["Study Type"].eq("INTERVENTIONAL")].copy()
    log.append(("Interventional only", len(df)))

    # model_text: four cleaned fields, combined once (no weighting)
    df["conditions_clean"] = _clean_text(df["Conditions"])
    df["interventions_clean"] = _clean_text(df["Interventions"])
    df["title_clean"] = _clean_text(df["Study Title"])
    df["summary_clean"] = _clean_text(df["Brief Summary"])
    df["model_text"] = (df["conditions_clean"] + " " + df["interventions_clean"] + " "
                        + df["title_clean"] + " " + df["summary_clean"]
                        ).str.replace(r"\s+", " ", regex=True).str.strip()
    df["text_length"] = df["model_text"].str.len()
    df = df.loc[df["text_length"].ge(25)].copy()
    log.append(("Model-ready (text ≥ 25 characters)", len(df)))

    # Display fields and recruitment flags
    df["Study URL"] = "https://clinicaltrials.gov/study/" + df["NCT Number"].astype(str)
    df["Sex"] = df["Sex"].astype("string").str.upper().str.strip().fillna("ALL")
    df["is_recruiting"] = df["Study Status"].eq("RECRUITING").fillna(False).astype(bool)
    df["is_upcoming"] = df["Study Status"].eq("NOT_YET_RECRUITING").fillna(False).astype(bool)
    log.append(("Recruiting (default search pool)", int(df["is_recruiting"].sum())))
    log.append(("Not yet recruiting", int(df["is_upcoming"].sum())))

    df["Eligibility_Display"] = eligibility_display.loc[df.index].fillna("").astype(str)
    df["Phase_Clean"] = df["Phases"].map(tidy_phase)
    df["Enrollment"] = pd.to_numeric(df["Enrollment"], errors="coerce")
    df["Country_List"] = df["Countries"].fillna("").astype(str).str.split("|").apply(
        lambda xs: [x.strip() for x in xs if x.strip()])
    for col in ["Sponsor", "Age", "Minimum Age", "Maximum Age", "Conditions", "Interventions", "Brief Summary"]:
        df[col] = df[col].fillna("").astype(str)
    return df.reset_index(drop=True), log


def split_data(prepared: pd.DataFrame):
    """80/20 train/test, then 75/25 inner-train/validation inside training (notebook cells 45–46)."""
    train_df, test_df = train_test_split(prepared, test_size=0.20, random_state=RANDOM_STATE)
    train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    inner_df, val_df = train_test_split(train_df, test_size=0.25, random_state=RANDOM_STATE)
    return train_df, test_df, inner_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# Stage 4 — Modelling
# ----------------------------------------------------------------------------------------
def new_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(stop_words="english", max_features=MAX_TFIDF_FEATURES, min_df=3, max_df=0.95,
                           ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)


def new_reducer(n_dim: int):
    return make_pipeline(TruncatedSVD(n_components=n_dim, n_iter=5, random_state=RANDOM_STATE),
                         Normalizer(copy=False))


def sampled_silhouette(X, labels, max_rows=500, seed=RANDOM_STATE) -> float:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(labels), min(max_rows, len(labels)), replace=False)
    if len(np.unique(labels[idx])) < 2:
        return np.nan
    return float(silhouette_score(X[idx], labels[idx], metric="euclidean"))


def retrieval_proxy_metrics(vec, reducer, km, x_pool, z_pool, pool_df, queries=EVALUATION_QUERIES, top_n=TOP_N):
    """Share of the full-cosine top-N kept by nearest-3-cluster preselection, and share of pool searched."""
    eligible = np.flatnonzero(pool_df["Study Status"].eq("RECRUITING").to_numpy())
    if len(eligible) < top_n:
        eligible = np.arange(len(pool_df))
    pool_labels = km.predict(z_pool[eligible])
    coverages, fractions = [], []
    for q in queries:
        q_tfidf = vec.transform([q])
        centre_scores = cosine_similarity(reducer.transform(q_tfidf), km.cluster_centers_).ravel()
        chosen = np.argsort(-centre_scores)[:min(N_NEAREST_CLUSTERS, km.n_clusters - 1)]
        mask = np.isin(pool_labels, chosen)
        direct = cosine_similarity(q_tfidf, x_pool[eligible]).ravel()
        top_local = np.argsort(-direct, kind="stable")[:min(top_n, len(direct))]
        coverages.append(float(np.mean(mask[top_local])))
        fractions.append(float(mask.sum() / len(eligible)))
    return float(np.mean(coverages)), float(np.mean(fractions))


def run_model_selection(inner_df, val_df, log=print) -> pd.DataFrame:
    """42-configuration experiment (6 SVD sizes × 7 K values) on inner-train / validation."""
    vec = new_vectorizer()
    x_in = vec.fit_transform(inner_df["model_text"])
    x_va = vec.transform(val_df["model_text"])
    rows = []
    for n_dim in SVD_DIMENSIONS:
        if n_dim >= min(x_in.shape):
            continue
        red = new_reducer(n_dim)
        z_in, z_va = red.fit_transform(x_in), red.transform(x_va)
        var = float(red.named_steps["truncatedsvd"].explained_variance_ratio_.sum())
        for k in K_VALUES:
            km = KMeans(n_clusters=k, init="k-means++", n_init=5, max_iter=250, random_state=RANDOM_STATE)
            lab_in = km.fit_predict(z_in)
            lab_va = km.predict(z_va)
            cov, frac = retrieval_proxy_metrics(vec, red, km, x_va, z_va, val_df)
            counts = np.bincount(lab_in, minlength=k)
            rows.append({"text_column": "model_text", "svd_dimensions": n_dim, "k": k,
                         "explained_variance": var, "inertia": float(km.inertia_),
                         "silhouette_train": sampled_silhouette(z_in, lab_in),
                         "silhouette_validation": sampled_silhouette(z_va, lab_va),
                         "validation_direct_top10_coverage": cov,
                         "validation_candidate_fraction": frac,
                         "smallest_cluster": int(counts.min()),
                         "largest_cluster_fraction": float(counts.max() / len(lab_in))})
            log(f"SVD={n_dim:<3} K={k:<2} coverage={cov:.3f} candidates={frac:.3f}")
    comp = pd.DataFrame(rows)
    comp["passes_retention"] = comp["explained_variance"].ge(0.10)
    comp["passes_balance"] = comp["largest_cluster_fraction"].le(0.70) & comp["smallest_cluster"].ge(25)
    comp["passes_coverage"] = comp["validation_direct_top10_coverage"].ge(0.90)
    comp["passes_all"] = comp[["passes_retention", "passes_balance", "passes_coverage"]].all(axis=1)
    comp["selection_score"] = (0.50 * comp["validation_direct_top10_coverage"]
                               + 0.20 * (1 - comp["validation_candidate_fraction"])
                               + 0.15 * comp["explained_variance"].clip(upper=1)
                               + 0.15 * ((comp["silhouette_validation"].fillna(-1) + 1) / 2))
    return comp


def select_configuration(comp: pd.DataFrame) -> pd.Series:
    """Same fallback order as notebook cell 56."""
    for mask in [comp["passes_all"], comp["passes_retention"] & comp["passes_balance"],
                 comp["passes_retention"], pd.Series(True, index=comp.index)]:
        pool = comp.loc[mask]
        if not pool.empty:
            return pool.sort_values(["selection_score", "validation_direct_top10_coverage", "svd_dimensions"],
                                    ascending=[False, False, False]).iloc[0]


@dataclass
class FinalModel:
    vectorizer: TfidfVectorizer
    reducer: object
    kmeans: KMeans
    svd_dimensions: int
    k: int
    explained_variance: float


def fit_final_model(train_df: pd.DataFrame, svd_dimensions: int, k: int) -> FinalModel:
    """Refit on the full 80% training split (notebook cells 61–63)."""
    vec = new_vectorizer()
    x_tr = vec.fit_transform(train_df["model_text"])
    red = new_reducer(svd_dimensions)
    z_tr = red.fit_transform(x_tr)
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=RANDOM_STATE)
    km.fit(z_tr)
    return FinalModel(vec, red, km, svd_dimensions, k,
                      float(red.named_steps["truncatedsvd"].explained_variance_ratio_.sum()))


def cluster_top_terms(model: FinalModel, n=12) -> dict[int, list[str]]:
    """Approximate centroid keywords via SVD components (notebook cell 68)."""
    terms = model.vectorizer.get_feature_names_out()
    approx = model.kmeans.cluster_centers_ @ model.reducer.named_steps["truncatedsvd"].components_
    return {c: list(terms[np.argsort(-approx[c])[:n]]) for c in range(model.k)}


# ----------------------------------------------------------------------------------------
# Stage 5 — Search (notebook cell 71, extended with display filters)
# ----------------------------------------------------------------------------------------
def rank_trial_search(query, pool_df, x_pool, z_pool, model: FinalModel, mask=None, top_n=TOP_N):
    """Return (full-cosine results, cluster-guided results, chosen clusters, centroid scores)."""
    eligible = np.flatnonzero(mask if mask is not None else pool_df["Study Status"].eq("RECRUITING").to_numpy())
    if len(eligible) == 0:
        raise ValueError("No studies match the selected filters.")
    q_tfidf = model.vectorizer.transform([query])
    if q_tfidf.nnz == 0:
        raise ValueError("None of those words are in the model vocabulary. Try other terms.")
    centre_scores = cosine_similarity(model.reducer.transform(q_tfidf), model.kmeans.cluster_centers_).ravel()
    chosen = np.argsort(-centre_scores)[:min(N_NEAREST_CLUSTERS, model.k - 1)]
    labels = model.kmeans.predict(z_pool[eligible])
    candidates = eligible[np.isin(labels, chosen)]
    s_full = cosine_similarity(q_tfidf, x_pool[eligible]).ravel()
    idx_full = eligible[np.argsort(-s_full, kind="stable")[:top_n]]
    sc_full = np.sort(s_full)[::-1][:top_n]
    if len(candidates):
        s_cl = cosine_similarity(q_tfidf, x_pool[candidates]).ravel()
        order = np.argsort(-s_cl, kind="stable")[:top_n]
        idx_cl, sc_cl = candidates[order], s_cl[order]
    else:
        idx_cl, sc_cl = np.array([], dtype=int), np.array([])

    def frame(idx, scores, how):
        out = pool_df.iloc[idx].copy()
        out.insert(0, "Rank", np.arange(1, len(out) + 1))
        out["Similarity"] = np.round(scores, 3)
        out["Method"] = how
        return out.reset_index(drop=True)

    info = {"eligible": int(len(eligible)), "candidates": int(len(candidates))}
    return frame(idx_full, sc_full, "full_cosine"), frame(idx_cl, sc_cl, "cluster_guided"), chosen, centre_scores, info
