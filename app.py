"""
Cancer Trial Discovery for Hospital Research Teams — Streamlit app
BSNS8030 · SDG 3 Good Health and Well-being · CRISP-DM Stage 6 (Deployment)

Search supports discovery for clinical review. It does not diagnose disease or
determine patient eligibility.

Run locally:   streamlit run app.py
Retrain/eval:  python train_model.py   (rewrites artifacts/)
"""
import json
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import pipeline as P

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ART = ROOT / "artifacts"
DATA_PATH = next((p for p in [DATA_DIR / "cancer_trials_10000.csv.gz", DATA_DIR / "cancer_trials_10000.csv"]
                  if p.exists()), None)

st.set_page_config(page_title="Cancer Trial Discovery · Hospital Review", page_icon="🏥", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg,#13344a 0%,#1f6f8b 100%); color:#fff; padding:1.3rem 1.6rem;
       border-radius:14px; margin-bottom:.9rem;}
.hero h1 {color:#fff; font-size:1.8rem; margin:0 0 .3rem 0;}
.hero p {margin:0; opacity:.92;}
.card {border:1px solid #dfe6ea; border-radius:12px; padding:.9rem 1.1rem; margin-bottom:.7rem; background:#fff;}
.card h4 {margin:.1rem 0 .3rem 0; font-size:1.02rem; color:#13344a;}
.pill {display:inline-block; padding:2px 9px; border-radius:999px; font-size:.75rem; margin:0 4px 4px 0;
       background:#edf4f7; color:#13344a; border:1px solid #d2e3ea;}
.pill.warn {background:#fff5e3; color:#8a5a00; border-color:#f0ddb2;}
.pill.miss {background:#fdecec; color:#9b2c2c; border-color:#f3c9c9;}
.score {float:right; font-weight:700; color:#1f6f8b;}
.muted {color:#56656d; font-size:.88rem;}
.notice {background:#fff8e6; border-left:4px solid #e0a526; padding:.65rem 1rem; border-radius:6px;
         font-size:.9rem; margin:.3rem 0 .9rem 0;}
</style>
""", unsafe_allow_html=True)

NOTICE = ("<div class='notice'><b>Discovery support, not clinical decision-making.</b> Results are ranked by text "
          "similarity to your search. They do not diagnose disease or determine patient eligibility, and the "
          "recruitment status comes from a historical snapshot. Verify every study in the live registry and "
          "review eligibility with the clinical team.</div>")


# ----------------------------------------------------------------------------------------
# Load data, artifacts and model (cached)
# ----------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_artifacts():
    summary = json.loads((ART / "model_summary.json").read_text())
    comp = pd.read_csv(ART / "model_selection_all_experiments.csv")
    clusters = pd.read_csv(ART / "clusters.csv")
    review = pd.read_csv(ART / "clinician_relevance_review_template.csv")
    return summary, comp, clusters, review


THEME_RULES = [  # readable names for exploratory clusters, matched on the cluster's top terms
    (("lung", "nsclc"), "Lung cancer"),
    (("breast", "her2"), "Breast cancer / HER2"),
    (("prostate",), "Prostate cancer"),
    (("leukemia", "lymphoma", "myeloma"), "Blood cancers"),
    (("solid tumors", "advanced solid"), "Advanced solid tumours (early-phase dosing)"),
    (("surgery", "behavioral", "care"), "Surgery, supportive care and behavioural"),
    (("different ways", "stop", "rationale"), "Mixed cancers, NCI-style summaries"),
    (("safety", "efficacy"), "Drug safety and efficacy (broad)"),
]


def theme_name(terms: list[str]) -> str:
    for keys, name in THEME_RULES:
        if any(k in terms[:6] for k in keys):
            return name
    return " / ".join(terms[:3]).title()


@st.cache_resource(show_spinner="Preparing data and fitting the selected model…")
def load_model():
    summary, *_ = load_artifacts()
    raw = pd.read_csv(DATA_PATH)
    prepared, log = P.prepare(raw)
    train_df, test_df, _, _ = P.split_data(prepared)
    model = P.fit_final_model(train_df, summary["svd_dimensions"], summary["k"])
    X_all = model.vectorizer.transform(prepared["model_text"])
    Z_all = model.reducer.transform(X_all)
    prepared["Cluster"] = model.kmeans.predict(Z_all)
    split = pd.Series("Train (80%)", index=prepared.index)
    split[prepared["NCT Number"].isin(set(test_df["NCT Number"]))] = "Test (20%)"
    prepared["Split"] = split
    terms = P.cluster_top_terms(model)
    themes = {c: theme_name(t) for c, t in terms.items()}
    prepared["Theme"] = prepared["Cluster"].map(themes)
    return prepared, log, model, X_all, Z_all, terms, themes


if DATA_PATH is None or not (ART / "model_summary.json").exists():
    st.error("Missing data/cancer_trials_10000.csv.gz or artifacts/. Run `python train_model.py` first.")
    st.stop()

summary, comp, clusters_art, review_template = load_artifacts()
T, prep_log, model, X_all, Z_all, TERMS, THEMES = load_model()
MODE = summary["search_mode"]
HT = summary["held_out_test"]

with st.sidebar:
    st.markdown("### 🏥 Cancer Trial Discovery")
    st.caption("For hospital clinical research teams")
    page = st.radio("Go to", ["Trial search", "Exploratory clusters", "Model evaluation",
                              "Data and preparation", "About and limitations"], label_visibility="collapsed")
    st.divider()
    st.markdown(f"**Model:** TF-IDF → SVD {summary['svd_dimensions']}D → K-means K={summary['k']}")
    st.markdown(f"**Adopted search:** `{MODE}`")
    st.caption("Data: ClinicalTrials.gov API v2 snapshot (historical). Recheck status in the live registry.")


def study_card(r, missing=False):
    countries = r["Country_List"]
    ctry = ", ".join(countries[:6]) + (f" +{len(countries) - 6} more" if len(countries) > 6 else "")
    enrol = f"{int(r['Enrollment']):,} planned" if pd.notna(r["Enrollment"]) else "Enrolment not recorded"
    status_cls = "" if r["Study Status"] == "RECRUITING" else "warn"
    extra = "<span class='pill miss'>Missed by cluster-guided search</span>" if missing else ""
    st.markdown(f"""
    <div class='card'><span class='score'>{r['Similarity']:.3f}</span>
      <h4>{r['Rank']}. {r['Study Title']}</h4>
      <span class='pill'>{r['NCT Number']}</span>
      <span class='pill {status_cls}'>{r['Study Status'].replace('_', ' ').title()}</span>
      <span class='pill'>{r['Phase_Clean']}</span>
      <span class='pill'>Cluster {r['Cluster']}: {r['Theme']}</span>{extra}
      <div class='muted' style='margin-top:.3rem'><b>Conditions:</b> {r['Conditions'][:220] or '—'}</div>
      <div class='muted'><b>Interventions:</b> {r['Interventions'][:220] or '—'}</div>
      <div class='muted'><b>Locations:</b> {ctry or 'Not listed'} · {enrol} · <b>Sponsor:</b> {r['Sponsor'] or '—'}</div>
    </div>""", unsafe_allow_html=True)
    with st.expander("Summary, eligibility criteria and registry link"):
        st.write(r["Brief Summary"] or "No summary provided.")
        st.caption(f"Ages: {r['Minimum Age'] or 'no minimum'} – {r['Maximum Age'] or 'no maximum'} · "
                   f"Sex: {r['Sex']} · Age groups: {r['Age'] or 'not stated'}")
        if r["Eligibility_Display"]:
            st.markdown("**Eligibility criteria as registered (for clinical review):**")
            st.text(r["Eligibility_Display"][:3000] + ("…" if len(r["Eligibility_Display"]) > 3000 else ""))
        st.link_button("Open live record on ClinicalTrials.gov ↗", r["Study URL"])


# ----------------------------------------------------------------------------------------
# Page 1 — Trial search (hospital review)
# ----------------------------------------------------------------------------------------
if page == "Trial search":
    st.markdown("""<div class='hero'><h1>Find cancer studies for clinical review</h1>
    <p>Search the registry snapshot by condition and intervention. The tool returns a ranked, manageable
    shortlist for your research team to review — it supports discovery, not diagnosis or eligibility.</p></div>""",
                unsafe_allow_html=True)
    st.markdown(NOTICE, unsafe_allow_html=True)

    query = st.text_input("Search by condition and intervention", "lung cancer immunotherapy",
                          help="e.g. 'breast cancer chemotherapy', 'glioblastoma temozolomide', 'liver cancer ablation'")
    all_countries = sorted({c for xs in T["Country_List"] for c in xs})
    c1, c2, c3, c4, c5 = st.columns([1.3, 1.2, 1.2, 1, 1])
    status_opt = c1.selectbox("Recruitment status", ["Recruiting", "Recruiting + not yet recruiting", "All statuses"])
    country = c2.selectbox("Site country", ["All"] + all_countries)
    phases = c3.multiselect("Phase", sorted(T["Phase_Clean"].unique()), placeholder="Any phase")
    age = c4.selectbox("Age group", ["Any", "CHILD", "ADULT", "OLDER_ADULT"],
                       format_func=lambda x: {"Any": "Any", "CHILD": "Child", "ADULT": "Adult", "OLDER_ADULT": "Older adult"}[x])
    sex = c5.selectbox("Sex", ["Any", "FEMALE", "MALE"], format_func=str.title)
    c6, c7 = st.columns([2, 1])
    view = c6.radio("Search method", ["Full cosine (adopted)", "Compare with cluster-guided"], horizontal=True,
                    help="Full cosine ranks every eligible study. Cluster-guided first keeps only the 3 nearest K-means "
                         "groups; it was tested but not adopted because it missed too many top results on held-out data.")
    top_n = c7.slider("Results to show", 5, 25, 10)

    mask = np.ones(len(T), dtype=bool)
    if status_opt == "Recruiting":
        mask &= T["is_recruiting"].to_numpy()
    elif status_opt.startswith("Recruiting +"):
        mask &= (T["is_recruiting"] | T["is_upcoming"]).to_numpy()
    if country != "All":
        mask &= T["Country_List"].apply(lambda xs: country in xs).to_numpy()
    if phases:
        mask &= T["Phase_Clean"].isin(phases).to_numpy()
    if age != "Any":
        mask &= T["Age"].str.contains(age, regex=False).to_numpy()
    if sex != "Any":
        mask &= T["Sex"].isin(["ALL", sex]).to_numpy()

    if query.strip():
        t0 = time.perf_counter()
        try:
            full, guided, chosen, centre_scores, info = P.rank_trial_search(query, T, X_all, Z_all, model, mask, top_n)
        except ValueError as e:
            st.warning(str(e))
            st.stop()
        ms = (time.perf_counter() - t0) * 1000
        full = full[full["Similarity"] > 0]
        guided = guided[guided["Similarity"] > 0]
        overlap = len(set(full["NCT Number"]) & set(guided["NCT Number"]))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Studies matching filters", f"{info['eligible']:,}")
        m2.metric("Results shown", len(full))
        m3.metric("Nearest clusters", ", ".join(str(c) for c in chosen))
        m4.metric("Search time", f"{ms:.0f} ms")

        if view.startswith("Full"):
            if full.empty:
                st.info("No studies matched. Widen the filters (e.g. All statuses or All countries) or use broader terms.")
            for _, r in full.iterrows():
                study_card(r)
        else:
            st.caption(f"Cluster-guided search kept {info['candidates']:,} of {info['eligible']:,} filtered studies "
                       f"({info['candidates'] / max(info['eligible'], 1):.0%}) and agreed on **{overlap} of {len(full)}** "
                       "top results with full cosine.")
            a, b = st.columns(2)
            a.markdown("##### Full cosine (adopted)")
            b.markdown("##### Cluster-guided (not adopted)")
            missed = set(full["NCT Number"]) - set(guided["NCT Number"])
            show_cols = ["Rank", "NCT Number", "Study Title", "Similarity", "Cluster"]
            a.dataframe(full[show_cols], hide_index=True, use_container_width=True)
            b.dataframe(guided[show_cols], hide_index=True, use_container_width=True)
            if missed:
                st.markdown(f"**{len(missed)} studies in the full-cosine list were not reached by cluster-guided search:**")
                for _, r in full[full["NCT Number"].isin(missed)].iterrows():
                    study_card(r, missing=True)

        sheet = full[["Rank", "NCT Number", "Study Title", "Study Status", "Phase_Clean", "Conditions",
                      "Interventions", "Countries", "Similarity", "Study URL"]].copy()
        sheet.insert(0, "Search query", query)
        sheet["Live status checked (Y/N)"] = ""
        sheet["Clinician relevance (0/1)"] = ""
        sheet["Reviewer notes"] = ""
        st.download_button("Download shortlist as a review worksheet (CSV)", sheet.to_csv(index=False),
                           file_name="trial_review_shortlist.csv")


# ----------------------------------------------------------------------------------------
# Page 2 — Exploratory clusters
# ----------------------------------------------------------------------------------------
elif page == "Exploratory clusters":
    st.title(f"Exploratory K-means groups (K = {summary['k']})")
    st.markdown(f"K-means grouped the studies in {summary['svd_dimensions']}-dimensional SVD space. These groups are "
                "**text patterns for exploring the registry**, not diagnoses, and they are **not used to filter the "
                "adopted search**. Names below are our interpretation of each group's top terms.")

    summ = (T.groupby("Cluster").agg(Studies=("NCT Number", "size"), Recruiting=("is_recruiting", "sum"),
                                     Median_enrolment=("Enrollment", "median")).reset_index())
    summ["Theme"] = summ["Cluster"].map(THEMES)
    summ = summ.merge(clusters_art[["cluster", "train_records", "test_records"]],
                      left_on="Cluster", right_on="cluster").drop(columns="cluster")
    summ["Share"] = summ["Studies"] / summ["Studies"].sum()

    st.altair_chart(alt.Chart(summ).mark_bar(color="#1f6f8b").encode(
        x=alt.X("Studies:Q"), y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=380)),
        tooltip=["Cluster", "Theme", "Studies", "Recruiting", alt.Tooltip("Share:Q", format=".0%")]
    ).transform_calculate(label="'Cluster ' + datum.Cluster + ' · ' + datum.Theme").properties(height=40 * len(summ)),
        use_container_width=True)
    big = summ.loc[summ["Share"].idxmax()]
    st.caption(f"Cluster {big['Cluster']} holds {big['Share']:.0%} of studies — a broad group, one reason the clusters "
               "are kept exploratory.")
    st.dataframe(summ[["Cluster", "Theme", "Studies", "Recruiting", "train_records", "test_records", "Median_enrolment"]]
                 .rename(columns={"train_records": "Train", "test_records": "Test", "Median_enrolment": "Median planned enrolment"}),
                 hide_index=True, use_container_width=True)

    st.subheader("Inspect a group")
    c = st.selectbox("Cluster", range(summary["k"]), format_func=lambda x: f"Cluster {x} · {THEMES[x]}")
    st.markdown(" ".join(f"<span class='pill'>{t}</span>" for t in TERMS[c]), unsafe_allow_html=True)
    st.caption("Approximate top terms from the centroid projected back through the SVD components (notebook cell 68).")
    members = T[T["Cluster"] == c]
    closeness = Z_all[members.index] @ (model.kmeans.cluster_centers_[c] / np.linalg.norm(model.kmeans.cluster_centers_[c]))
    rep = members.assign(Closeness=np.round(closeness, 3)).sort_values("Closeness", ascending=False)
    st.dataframe(rep[["NCT Number", "Study Title", "Study Status", "Phase_Clean", "Closeness", "Study URL"]].head(25),
                 hide_index=True, use_container_width=True,
                 column_config={"Study URL": st.column_config.LinkColumn("Registry", display_text="Open ↗")})


# ----------------------------------------------------------------------------------------
# Page 3 — Model evaluation
# ----------------------------------------------------------------------------------------
elif page == "Model evaluation":
    st.title("Model selection and evaluation")
    r = summary["records"]
    st.markdown(f"TF-IDF ({summary['tfidf_features']:,} features) → Truncated SVD + Normalizer → K-means. "
                f"42 configurations were compared on the inner training set ({r['inner_train']:,}) and validation set "
                f"({r['validation']:,}); the selected model was refitted on the full training split ({r['train']:,}) "
                f"and evaluated once on the untouched test set ({r['test']:,}).")

    st.subheader("1 · Decision on held-out test data")
    thr = summary["thresholds"]
    a, b, c = st.columns(3)
    cov_ok = HT["top10_coverage"] >= thr["coverage_min"]
    frac_ok = HT["candidate_fraction"] <= thr["candidate_fraction_max"]
    a.metric("Top-10 coverage vs full cosine", f"{HT['top10_coverage']:.2%}")
    a.caption(f"Target ≥ {thr['coverage_min']:.0%} — {'met' if cov_ok else 'not met'}")
    b.metric("Share of recruiting studies searched", f"{HT['candidate_fraction']:.2%}")
    b.caption(f"Target ≤ {thr['candidate_fraction_max']:.0%} — {'met' if frac_ok else 'not met'}")
    c.metric("Adopted search mode", MODE)
    c.caption("Cluster-guided only if both targets are met")
    st.info("Cluster-guided search had to keep at least 95% of the full-cosine top 10 while searching no more than 70% "
            "of recruiting studies. It did not meet both targets, so the prototype uses **full cosine ranking**; "
            "K-means remains an exploratory view. Coverage is agreement with an algorithmic baseline, not clinical accuracy.")

    st.subheader("2 · Selected configuration")
    v = summary["validation"]
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("SVD dimensions", summary["svd_dimensions"])
    x2.metric("Clusters (K)", summary["k"])
    x3.metric("TF-IDF variance retained", f"{summary['explained_variance']:.2%}")
    x4.metric("Validation top-10 coverage", f"{v['top10_coverage']:.2%}")
    y1, y2, y3 = st.columns(3)
    y1.metric("Validation share searched", f"{v['candidate_fraction']:.2%}")
    y2.metric("Train silhouette", f"{HT['train_silhouette']:.4f}")
    y3.metric("Test silhouette", f"{HT['test_silhouette']:.4f}")
    if summary.get("configuration_source") == "presented":
        rc = summary["rule_choice_on_this_snapshot"]
        st.caption(f"Configuration as presented (100D, K=6, selected on the notebook's own API snapshot of 7,873 studies). "
                   f"On this CSV snapshot ({r['prepared']:,} studies) the same weighted rule would narrowly prefer "
                   f"{rc['svd_dimensions']}D, K={rc['k']}; the adopted search mode is full cosine either way.")

    st.subheader("3 · All 42 configurations (validation)")
    metric = st.selectbox("Colour by", ["selection_score", "validation_direct_top10_coverage",
                                        "validation_candidate_fraction", "silhouette_validation", "explained_variance"])
    heat = comp.copy()
    heat["selected"] = (heat["svd_dimensions"] == summary["svd_dimensions"]) & (heat["k"] == summary["k"])
    base = alt.Chart(heat).encode(x=alt.X("k:O", title="Clusters (K)", axis=alt.Axis(labelAngle=0)),
                                  y=alt.Y("svd_dimensions:O", title="SVD dimensions", sort="descending"))
    st.altair_chart((base.mark_rect().encode(color=alt.Color(f"{metric}:Q", scale=alt.Scale(scheme="tealblues")),
                                             tooltip=["svd_dimensions", "k", alt.Tooltip(metric, format=".3f"),
                                                      "passes_all"])
                     + base.mark_text(fontSize=11).encode(text=alt.Text(f"{metric}:Q", format=".2f"),
                                                          color=alt.condition("datum.selected", alt.value("#c0392b"), alt.value("#13344a"))))
                    .properties(height=300), use_container_width=True)
    st.caption("Selection score = 50% top-10 coverage + 20% candidate reduction + 15% SVD variance + 15% silhouette. "
               "Red text marks the configuration used.")

    st.subheader(f"4 · Supporting diagnostics at {summary['svd_dimensions']}D")
    sel = comp[comp["svd_dimensions"] == summary["svd_dimensions"]].sort_values("k")
    l, rr = st.columns(2)
    l.altair_chart(alt.Chart(sel).mark_line(point=True, color="#13344a").encode(
        x=alt.X("k:O", axis=alt.Axis(labelAngle=0)), y=alt.Y("inertia:Q", scale=alt.Scale(zero=False)))
        .properties(title="Elbow (inertia)"), use_container_width=True)
    sil = sel.melt(id_vars="k", value_vars=["silhouette_train", "silhouette_validation"], var_name="Set", value_name="Silhouette")
    rr.altair_chart(alt.Chart(sil).mark_line(point=True).encode(
        x=alt.X("k:O", axis=alt.Axis(labelAngle=0)), y="Silhouette:Q",
        color=alt.Color("Set:N", scale=alt.Scale(range=["#1f6f8b", "#e0a526"]), legend=alt.Legend(orient="bottom")))
        .properties(title="Silhouette: train vs validation"), use_container_width=True)
    cov = sel.melt(id_vars="k", value_vars=["validation_direct_top10_coverage", "validation_candidate_fraction"],
                   var_name="Measure", value_name="Fraction")
    st.altair_chart(alt.Chart(cov).mark_line(point=True).encode(
        x=alt.X("k:O", axis=alt.Axis(labelAngle=0)), y=alt.Y("Fraction:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Measure:N", scale=alt.Scale(range=["#1f6f8b", "#c0392b"]), legend=alt.Legend(orient="bottom")))
        .properties(title="Validation: top-10 coverage vs share searched"), use_container_width=True)
    st.caption("Inertia falls smoothly with K, so the elbow does not decide K on its own. Silhouette stays low: "
               "weak geometric separation, not a relevance score.")

    st.subheader("5 · Clinical relevance (Precision@10)")
    st.markdown("Precision@10 is **not calculated**: it needs independent clinician relevance labels. The review "
                "template lists the top 10 held-out results for 12 queries under both methods, ready for labelling.")
    st.download_button("Download clinician review template (CSV)", review_template.to_csv(index=False),
                       file_name="clinician_relevance_review_template.csv")


# ----------------------------------------------------------------------------------------
# Page 4 — Data and preparation
# ----------------------------------------------------------------------------------------
elif page == "Data and preparation":
    st.title("Data understanding and preparation")
    st.dataframe(pd.DataFrame(prep_log, columns=["Preparation milestone", "Studies"]), hide_index=True)
    r = summary["records"]
    st.markdown("**model_text** = cleaned conditions + interventions + study title + brief summary, combined once "
                "(lower-cased, whitespace normalised, 'tumour' → 'tumor'). No field is weighted or repeated. "
                "NCT ID, status, location and eligibility stay as metadata for filters and display.")
    split = pd.DataFrame({"Partition": ["Inner training (fit)", "Validation (model selection)", "Untouched test"],
                          "Studies": [r["inner_train"], r["validation"], r["test"]]})
    st.altair_chart(alt.Chart(split).mark_bar().encode(
        x=alt.X("sum(Studies):Q", stack="normalize", title="Share of model-ready studies", axis=alt.Axis(format="%")),
        color=alt.Color("Partition:N", sort=list(split["Partition"]),
                        scale=alt.Scale(range=["#1f6f8b", "#7fb3c8", "#e0a526"]), legend=alt.Legend(orient="bottom")),
        tooltip=["Partition", "Studies"]).properties(height=90, title="60 / 20 / 20 split (final model refits on train = inner + validation)"),
        use_container_width=True)

    l, rr = st.columns(2)
    stc = T["Study Status"].value_counts().rename_axis("Status").reset_index(name="Studies")
    l.altair_chart(alt.Chart(stc).mark_bar().encode(
        x="Studies:Q", y=alt.Y("Status:N", sort="-x", title=None),
        color=alt.condition(alt.datum.Status == "RECRUITING", alt.value("#1f6f8b"), alt.value("#c7d1d6")))
        .properties(title="Status of model-ready studies"), use_container_width=True)
    ph = T["Phase_Clean"].value_counts().rename_axis("Phase").reset_index(name="Studies")
    rr.altair_chart(alt.Chart(ph).mark_bar(color="#13344a").encode(
        x="Studies:Q", y=alt.Y("Phase:N", sort="-x", title=None)).properties(title="Phase"), use_container_width=True)
    ct = T["Country_List"].explode().value_counts().head(15).rename_axis("Country").reset_index(name="Studies")
    st.altair_chart(alt.Chart(ct).mark_bar(color="#1f6f8b").encode(
        x="Studies:Q", y=alt.Y("Country:N", sort="-x", title=None))
        .properties(title="Most listed site countries (a study can list several)"), use_container_width=True)


# ----------------------------------------------------------------------------------------
# Page 5 — About and limitations
# ----------------------------------------------------------------------------------------
else:
    st.title("About this prototype")
    st.markdown(f"""
**Purpose.** Help hospital clinical research teams find potentially relevant cancer studies faster, by returning a
ranked, manageable shortlist from thousands of registry records. Discovery supports — but does not replace — clinical review.

**Users.** Clinical research coordinators, research nurses, oncologists and trial units.

**How it works.** Each study's conditions, interventions, title and summary form one `model_text`. TF-IDF turns it into
{summary['tfidf_features']:,} word and phrase features. A search is transformed with the same vocabulary and studies are
ranked by cosine similarity (**full cosine**, the adopted mode). K-means (K={summary['k']}) on {summary['svd_dimensions']}-dimensional
SVD vectors provides exploratory groups and was tested as a pre-filter, but not adopted.

**Limitations**
- **Clinical relevance:** Precision@10 was not calculated — no independent clinician relevance labels yet.
- **Historical snapshot:** recruitment status must be rechecked in the live registry.
- **Eligibility:** ranked studies do not establish patient eligibility; eligibility is a clinical decision.
- **Coverage metric:** top-10 coverage measures overlap with full-cosine ranking, not clinical accuracy.
- **Weak cluster separation:** silhouette ≈ {HT['test_silhouette']:.2f}; clusters are text patterns, not diagnoses.

**Potential implications (not measured outcomes).** Surfacing research opportunities, reducing search effort for
clinicians, and supporting informed conversations about research options — in line with SDG 3.

**Sources.** ClinicalTrials.gov API v2 · scikit-learn (TfidfVectorizer, TruncatedSVD, KMeans, cosine similarity) ·
group Jupyter notebook (CRISP-DM).
""")
