"""
Modelling pipeline for the Cancer Clinical Trial Discovery app (BSNS8030, SDG 3).

CRISP-DM stages 3-4:
  Data preparation -> TF-IDF -> Truncated SVD -> K-means -> cosine-similarity retrieval

Retrieval pipeline (patient search):
  1. Query -> fitted TF-IDF vectoriser -> fitted SVD (same feature space as K-means)
  2. Cosine similarity between query and every K-means centroid -> nearest clusters
  3. Cosine similarity between query and each trial inside those clusters -> ranking
  4. Optional "safety net": add high-scoring trials from the whole catalogue so that
     overlapping clusters do not hide relevant studies
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

RANDOM_STATE = 42

# Words that appear in nearly every cancer trial; they add little to cluster themes.
DOMAIN_STOP_WORDS = {
    "cancer", "cancers", "study", "studies", "patients", "patient", "trial", "trials",
    "treatment", "treat", "treating", "participants", "phase", "clinical", "research",
    "purpose", "aim", "aims", "researchers", "learn", "people", "evaluate", "investigate",
    # Intervention-type prefixes from the API export ("DRUG: x | PROCEDURE: y")
    "drug", "drugs", "procedure", "device", "behavioral", "biological", "radiation", "other",
    "diagnostic_test", "dietary_supplement", "genetic", "combination_product",
    # NCI boilerplate repeated across hundreds of summaries
    "ways", "stop", "stopping", "dividing", "killing", "different", "spreading", "work",
}
STOP_WORDS = list(ENGLISH_STOP_WORDS.union(DOMAIN_STOP_WORDS))

# Column names exactly as produced by the group notebook (latest.ipynb, api_record_to_row).
SOURCE_COLUMNS = ["NCT Number", "Study Title", "Study URL", "Study Status", "Brief Summary", "Conditions",
                  "Interventions", "Sponsor", "Collaborators", "Sex", "Age", "Study Type", "Phases",
                  "Enrollment", "Countries", "Eligibility Criteria", "Minimum Age", "Maximum Age"]
REQUIRED = ["NCT Number", "Study Title", "Study Status", "Brief Summary", "Conditions", "Interventions",
            "Study Type", "Countries"]


def normalise_text(text: str) -> str:
    """Used for search queries so they match the cleaned model text."""
    text = str(text).lower()
    text = re.sub(r"\btumours?\b", "tumor", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tidy_phase(value: str) -> str:
    v = str(value or "").upper()
    if "EARLY_PHASE1" in v:
        return "Early Phase 1"
    nums = re.findall(r"PHASE(\d)", v)
    return "Phase " + "/".join(sorted(set(nums))) if nums else "Not recorded"


def _clean(series: pd.Series) -> pd.Series:
    """Notebook cells 36-39: lower-case and standardise tumour/tumours -> tumor."""
    return series.fillna("").astype(str).str.lower().str.replace(r"\btumours?\b", "tumor", regex=True)


# --------------------------------------------------------------------------------------
# Stage 3: Data preparation (mirrors latest.ipynb cells 27-54)
# --------------------------------------------------------------------------------------
def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    log = {"Raw records": len(raw)}
    df = raw.copy()
    for col in SOURCE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Cell 28: normalise whitespace in every text column
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x)
    eligibility_raw = raw["Eligibility Criteria"] if "Eligibility Criteria" in raw.columns else pd.Series("", index=raw.index)

    # Cells 29-31: valid NCT IDs, remove duplicates
    df["NCT Number"] = df["NCT Number"].astype("string").str.upper().str.strip()
    df = df[df["NCT Number"].str.fullmatch(r"NCT\d{8}", na=False)]
    df = df.drop_duplicates(subset=["NCT Number"], keep="first")
    log["Valid NCT ID + deduplicated"] = len(df)

    # Cells 32-34: standardise type/status, keep interventional studies
    df["Study Type"] = df["Study Type"].astype("string").str.upper().str.replace(" ", "_", regex=False)
    df["Study Status"] = df["Study Status"].astype("string").str.upper().str.replace(" ", "_", regex=False)
    df = df[df["Study Type"].eq("INTERVENTIONAL")].copy()
    log["Interventional only"] = len(df)

    # Cells 36-41: cleaned text fields and the weighted model text
    df["conditions_clean"] = _clean(df["Conditions"])
    df["interventions_clean"] = _clean(df["Interventions"])
    df["title_clean"] = _clean(df["Study Title"])
    df["summary_clean"] = _clean(df["Brief Summary"])
    df["model_text"] = (df["conditions_clean"] + " " + df["interventions_clean"] + " " + df["title_clean"]
                        + " " + df["summary_clean"]).str.replace(r"\s+", " ", regex=True).str.strip()
    df["weighted_model_text"] = (
        df["conditions_clean"] + " " + df["conditions_clean"] + " " + df["interventions_clean"] + " "
        + df["interventions_clean"] + " " + df["title_clean"] + " " + df["summary_clean"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # Cells 42-43: drop trials with < 25 characters of text
    df = df[df["model_text"].str.len() >= 25].copy()
    log["Text-ready interventional"] = len(df)

    # Cells 44-47: URL, sex, recruiting flags
    df["Study URL"] = "https://clinicaltrials.gov/study/" + df["NCT Number"].astype(str)
    df["Sex"] = df["Sex"].astype("string").str.upper().str.strip().fillna("ALL")
    df["is_recruiting"] = df["Study Status"].eq("RECRUITING").fillna(False).astype(bool)
    df["is_upcoming"] = df["Study Status"].eq("NOT_YET_RECRUITING").fillna(False).astype(bool)
    log["Recruiting (search catalogue)"] = int(df["is_recruiting"].sum())
    log["Not yet recruiting"] = int(df["is_upcoming"].sum())

    # Display-friendly internal columns used by the app
    df = df.reset_index(drop=False).rename(columns={"index": "_raw_index"})
    df["NCT_ID"] = df["NCT Number"].astype(str)
    df["Title"] = df["Study Title"].fillna("").astype(str)
    df["Summary"] = df["Brief Summary"].fillna("").astype(str)
    df["Conditions"] = df["Conditions"].fillna("").astype(str).str.replace(" | ", "; ", regex=False)
    df["Interventions"] = df["Interventions"].fillna("").astype(str).str.replace(" | ", "; ", regex=False)
    df["Countries"] = df["Countries"].fillna("").astype(str)
    df["Country_List"] = df["Countries"].str.split("|").apply(lambda xs: [x.strip() for x in xs if x.strip()])
    df["Eligibility_Criteria"] = eligibility_raw.loc[df["_raw_index"]].fillna("").astype(str).to_numpy()
    df["Study_URL"] = df["Study URL"]
    df["Status"] = df["Study Status"].astype(str)
    df["Phase_Clean"] = df["Phases"].map(tidy_phase)
    df["Enrollment"] = pd.to_numeric(df["Enrollment"], errors="coerce")
    df["Min_Age"] = df["Minimum Age"].fillna("").astype(str)
    df["Max_Age"] = df["Maximum Age"].fillna("").astype(str)
    df["Age_Groups"] = df["Age"].fillna("").astype(str)
    df["Sponsor"] = df["Sponsor"].fillna("").astype(str)
    df["Search_Text"] = df["weighted_model_text"]
    return df, log


# --------------------------------------------------------------------------------------
# Stage 4: Modelling
# --------------------------------------------------------------------------------------
@dataclass
class TrialModel:
    trials: pd.DataFrame
    vectorizer: TfidfVectorizer
    X_tfidf: object          # sparse, L2-normalised
    svd: TruncatedSVD
    X_svd: np.ndarray        # L2-normalised SVD features (used by K-means)
    kmeans: KMeans
    centroids: np.ndarray    # L2-normalised centroids in SVD space
    cluster_terms: dict
    cluster_labels: dict
    k: int
    svd_components: int
    explained_variance: float

    # ----------------------------------------------------------------------------------
    def query_vectors(self, query: str):
        q = normalise_text(query)
        if not q:
            raise ValueError("Please enter a search query.")
        q_tfidf = self.vectorizer.transform([q])
        if q_tfidf.nnz == 0:
            raise ValueError("None of those words appear in the trial catalogue. Try different terms.")
        q_svd = normalize(self.svd.transform(q_tfidf))
        return q_tfidf, q_svd

    def centroid_similarity(self, query: str) -> pd.DataFrame:
        _, q_svd = self.query_vectors(query)
        sims = cosine_similarity(q_svd, self.centroids).ravel()
        out = pd.DataFrame({
            "Cluster": np.arange(self.k),
            "Theme": [self.cluster_labels[c] for c in range(self.k)],
            "Centroid_Similarity": sims,
            "Trials": np.bincount(self.trials["Cluster"], minlength=self.k),
        })
        return out.sort_values("Centroid_Similarity", ascending=False).reset_index(drop=True)

    def search(self, query: str, method: str = "cluster_safety", n_clusters: int = 2,
               top_n: int = 10, country: str = "All", phases: list | None = None,
               safety_n: int = 3, min_similarity: float = 0.05, status: str = "recruiting",
               age_group: str = "Any", sex: str = "Any") -> tuple[pd.DataFrame, dict]:
        """method: 'global' | 'cluster' | 'cluster_safety'
        status: 'recruiting' (default patient catalogue) | 'recruiting_upcoming' | 'all'"""
        t0 = time.perf_counter()
        q_tfidf, _ = self.query_vectors(query)
        cent = self.centroid_similarity(query)
        chosen = cent["Cluster"].head(n_clusters).tolist()

        mask = np.ones(len(self.trials), dtype=bool)
        if status == "recruiting":
            mask &= self.trials["is_recruiting"].to_numpy()
        elif status == "recruiting_upcoming":
            mask &= (self.trials["is_recruiting"] | self.trials["is_upcoming"]).to_numpy()
        if age_group and age_group != "Any":
            mask &= self.trials["Age_Groups"].str.contains(age_group, regex=False).to_numpy()
        if sex and sex != "Any":
            mask &= self.trials["Sex"].isin(["ALL", sex]).to_numpy()
        if country and country != "All":
            mask &= self.trials["Country_List"].apply(lambda xs: country in xs).to_numpy()
        if phases:
            mask &= self.trials["Phase_Clean"].isin(phases).to_numpy()

        global_scores = cosine_similarity(q_tfidf, self.X_tfidf).ravel()

        if method == "global":
            cand = np.flatnonzero(mask)
        else:
            cand = np.flatnonzero(mask & self.trials["Cluster"].isin(chosen).to_numpy())

        order = cand[np.argsort(-global_scores[cand], kind="stable")]
        order = order[global_scores[order] >= min_similarity]
        top = list(order[:top_n])
        source = {i: ("Global search" if method == "global" else "Matched cluster") for i in top}

        if method == "cluster_safety":
            outside = np.flatnonzero(mask & ~self.trials["Cluster"].isin(chosen).to_numpy())
            outside = outside[np.argsort(-global_scores[outside], kind="stable")][:safety_n]
            outside = outside[global_scores[outside] >= min_similarity]
            # Add outside trials only when they score at least as well as the weakest in-cluster result.
            floor = global_scores[top[-1]] if len(top) == top_n else 0
            for i in outside:
                if global_scores[i] >= floor * 0.9 and global_scores[i] > 0:
                    top.append(i)
                    source[i] = "Safety net (other cluster)"
            top = sorted(top, key=lambda i: -global_scores[i])[:top_n]

        res = self.trials.iloc[top].copy()
        res.insert(0, "Rank", range(1, len(res) + 1))
        res["Similarity"] = global_scores[top].round(3)
        res["Found_Via"] = [source[i] for i in top]
        res["Theme"] = res["Cluster"].map(self.cluster_labels)
        info = {
            "clusters_searched": chosen if method != "global" else "all",
            "candidates_scored": int(len(cand)),
            "catalogue_size": int(mask.sum()),
            "seconds": time.perf_counter() - t0,
        }
        return res.reset_index(drop=True), info


def _cluster_labels(trials: pd.DataFrame, X_tfidf, terms, k: int):
    """Top terms per cluster from the mean TF-IDF vector of member trials."""
    cluster_terms, labels = {}, {}
    for c in range(k):
        idx = np.flatnonzero(trials["Cluster"].to_numpy() == c)
        mean_vec = np.asarray(X_tfidf[idx].mean(axis=0)).ravel()
        top = [terms[i] for i in np.argsort(mean_vec)[::-1][:12]]
        cluster_terms[c] = top
        # Readable label: first 3 distinct terms, skipping unigrams already inside a chosen bigram.
        picked = []
        for t in top:
            if any(t in p.split() or p in t.split() for p in picked):
                continue
            picked.append(t)
            if len(picked) == 3:
                break
        labels[c] = " / ".join(p.title() for p in picked)
    return cluster_terms, labels


def build_model(trials: pd.DataFrame, k: int = 8, svd_components: int = 80,
                test_size: float = 0.20) -> TrialModel:
    """Fit TF-IDF, SVD and K-means on the 80% training split (same split as latest.ipynb cell 53),
    then transform every interventional trial and assign it to its nearest centroid."""
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(np.arange(len(trials)), test_size=test_size, random_state=RANDOM_STATE)
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, min_df=2, max_df=0.95,
                                 max_features=5000, ngram_range=(1, 2), sublinear_tf=True, norm="l2")
    vectorizer.fit(trials["Search_Text"].iloc[train_idx])
    X = vectorizer.transform(trials["Search_Text"])
    n_comp = min(svd_components, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    svd.fit(X[train_idx])
    X_svd = normalize(svd.transform(X))
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
    km.fit(X_svd[train_idx])
    trials = trials.copy()
    trials["Cluster"] = km.predict(X_svd)
    trials["Split"] = "Train"
    trials.loc[trials.index[test_idx], "Split"] = "Test"
    terms = vectorizer.get_feature_names_out()
    cluster_terms, labels = _cluster_labels(trials, X, terms, k)
    return TrialModel(trials, vectorizer, X, svd, X_svd, km, normalize(km.cluster_centers_),
                      cluster_terms, labels, k, n_comp, float(svd.explained_variance_ratio_.sum()))


# --------------------------------------------------------------------------------------
# Stage 5: Evaluation helpers
# --------------------------------------------------------------------------------------
def k_selection(X_svd: np.ndarray, split: np.ndarray, k_values=range(3, 13)) -> pd.DataFrame:
    """Fit on training rows; report inertia plus cosine silhouette on train and on unseen test rows."""
    tr, te = split == "Train", split == "Test"
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300).fit(X_svd[tr])
        lab_tr, lab_te = km.labels_, km.predict(X_svd[te])
        rows.append({"K": k, "Inertia": km.inertia_,
                     "Train silhouette": silhouette_score(X_svd[tr], lab_tr, metric="cosine",
                                                          sample_size=min(3000, tr.sum()), random_state=RANDOM_STATE),
                     "Test silhouette": silhouette_score(X_svd[te], lab_te, metric="cosine")})
    return pd.DataFrame(rows)


def stability(X_svd: np.ndarray, k: int, seeds=(1, 2, 3, 4, 5)) -> pd.DataFrame:
    from sklearn.metrics import adjusted_rand_score
    base = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X_svd)
    rows = []
    for s in seeds:
        lab = KMeans(n_clusters=k, random_state=s, n_init=10).fit_predict(X_svd)
        rows.append({"Seed": s, "Adjusted_Rand_Index": adjusted_rand_score(base, lab)})
    return pd.DataFrame(rows)


EVALUATION_QUERIES = {
    "Q01": "non small cell lung cancer immunotherapy",
    "Q02": "HER2 positive breast cancer targeted therapy",
    "Q03": "metastatic colorectal cancer chemotherapy",
    "Q04": "prostate cancer hormone therapy",
    "Q05": "melanoma checkpoint inhibitor",
    "Q06": "ovarian cancer PARP inhibitor",
    "Q07": "lymphoma CAR T cell therapy",
    "Q08": "multiple myeloma maintenance treatment",
    "Q09": "pancreatic cancer surgery",
    "Q10": "exercise quality of life breast cancer survivors",
}


def compare_methods(model: TrialModel, queries=EVALUATION_QUERIES, top_n=10, n_clusters=2):
    """Agreement of cluster-assisted search with global search (proxy for coverage, NOT relevance)."""
    rows = []
    for qid, q in queries.items():
        g, gi = model.search(q, "global", top_n=top_n, min_similarity=0)
        gset = set(g["NCT_ID"])
        for m in ("cluster", "cluster_safety"):
            r, ri = model.search(q, m, n_clusters=n_clusters, top_n=top_n, min_similarity=0)
            rows.append({
                "Query_ID": qid, "Query": q, "Method": m,
                "Overlap_with_global_top10": len(gset & set(r["NCT_ID"])) / max(len(gset), 1),
                "Candidates_scored": ri["candidates_scored"],
                "Search_space_reduction": 1 - ri["candidates_scored"] / gi["candidates_scored"],
            })
    return pd.DataFrame(rows)
