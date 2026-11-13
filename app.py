"""
Cancer Clinical Trial Discovery — Streamlit prototype
BSNS8030 · SDG 3 Good Health and Well-being · CRISP-DM Stage 6 (Deployment)

Run:  streamlit run app.py
Data: data/cancer_trials_10000.csv  (exported by notebooks/latest.ipynb from the ClinicalTrials.gov API)
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from model import EVALUATION_QUERIES, build_model, compare_methods, k_selection, prepare, stability

st.set_page_config(page_title="Cancer Trial Finder", page_icon="🔬", layout="wide")
DATA_PATH = Path(__file__).parent / "data" / "cancer_trials_10000.csv"

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg,#0f4c5c 0%,#1b7f8c 100%); color:#fff; padding:1.4rem 1.6rem;
       border-radius:14px; margin-bottom:1rem;}
.hero h1 {color:#fff; font-size:1.9rem; margin:0 0 .3rem 0;}
.hero p {margin:0; opacity:.92;}
.card {border:1px solid #e3e8ec; border-radius:12px; padding:1rem 1.1rem; margin-bottom:.8rem; background:#fff;}
.card h4 {margin:.1rem 0 .35rem 0; font-size:1.05rem; color:#0f4c5c;}
.pill {display:inline-block; padding:2px 9px; border-radius:999px; font-size:.75rem; margin:0 4px 4px 0;
       background:#eef5f6; color:#0f4c5c; border:1px solid #d3e6e8;}
.pill.warn {background:#fff6e5; color:#8a5a00; border-color:#f1dfb5;}
.pill.net {background:#f3edff; color:#5a3d9a; border-color:#ddd0f7;}
.score {float:right; font-weight:700; color:#1b7f8c; font-size:1.05rem;}
.muted {color:#5b6b75; font-size:.88rem;}
.notice {background:#fff8e6; border-left:4px solid #f0b429; padding:.7rem 1rem; border-radius:6px;
         font-size:.9rem; margin:.4rem 0 1rem 0;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Data + model (cached)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw(file_bytes: bytes | None):
    if file_bytes is not None:
        from io import BytesIO
        return pd.read_csv(BytesIO(file_bytes))
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Training TF-IDF → SVD → K-means…")
def get_model(file_bytes: bytes | None, k: int, svd_components: int):
    raw = load_raw(file_bytes)
    trials, log = prepare(raw)
    return build_model(trials, k=k, svd_components=svd_components), log


with st.sidebar:
    st.markdown("### 🔬 Cancer Trial Finder")
    page = st.radio("Go to", ["Find trials", "How the search works", "Explore research themes",
                              "Data overview", "Model evaluation"], label_visibility="collapsed")
    st.divider()
    with st.expander("Model settings", expanded=False):
        k = st.slider("Number of clusters (K)", 4, 14, 8)
        svd_components = st.select_slider("SVD components", [40, 60, 80, 100, 150], value=80)
        uploaded = st.file_uploader("Use a different ClinicalTrials.gov CSV", type="csv")
    st.caption("Data: ClinicalTrials.gov snapshot. Statuses change; always confirm on the official record.")

file_bytes = uploaded.getvalue() if uploaded else None
if file_bytes is None and not DATA_PATH.exists():
    st.error(f"Put cancer_trials_10000.csv in {DATA_PATH.parent} or upload one in the sidebar.")
    st.stop()

model, prep_log = get_model(file_bytes, k, svd_components)
T = model.trials

DISCLAIMER = ("<div class='notice'><b>This tool helps you discover trials to discuss with your cancer care team.</b> "
              "It ranks studies by how closely their descriptions match your words. It does <b>not</b> decide "
              "whether you are eligible, and it is not medical advice.</div>")


def trial_card(r):
    via = r["Found_Via"]
    via_cls = "net" if via.startswith("Safety") else ""
    countries = r["Country_List"]
    country_txt = ", ".join(countries[:6]) + (f" +{len(countries) - 6} more" if len(countries) > 6 else "")
    enrol = f"{int(r['Enrollment']):,} planned participants" if pd.notna(r["Enrollment"]) else "Enrolment not recorded"
    st.markdown(f"""
    <div class='card'>
      <span class='score'>{r['Similarity']:.2f}</span>
      <h4>{r['Rank']}. {r['Title']}</h4>
      <span class='pill'>{r['NCT_ID']}</span>
      <span class='pill {"" if r["Status"] == "RECRUITING" else "warn"}'>{r['Status'].replace('_', ' ').title()}</span>
      <span class='pill'>{r['Phase_Clean']}</span>
      <span class='pill'>Cluster {r['Cluster']}: {r['Theme']}</span>
      <span class='pill {via_cls}'>{via}</span>
      <div class='muted' style='margin-top:.35rem'><b>Condition:</b> {r['Conditions'][:220] or '—'}</div>
      <div class='muted'><b>Treatment studied:</b> {r['Interventions'][:220] or '—'}</div>
      <div class='muted'><b>Locations listed:</b> {country_txt or 'Not listed'} · {enrol}</div>
      <div class='muted'><b>Sponsor:</b> {r['Sponsor'] or '—'}</div>
    </div>""", unsafe_allow_html=True)
    with st.expander("Summary, eligibility criteria and official link"):
        st.write(r["Summary"] or "No summary provided.")
        if r.get("Min_Age") or r.get("Max_Age") or r.get("Sex"):
            st.caption(f"Age: {r.get('Min_Age') or 'no minimum'} – {r.get('Max_Age') or 'no maximum'} · Sex: {r.get('Sex') or 'not stated'}")
        if r["Eligibility_Criteria"]:
            st.markdown("**Eligibility criteria (as registered — your doctor must confirm):**")
            st.text(r["Eligibility_Criteria"][:2500] + ("…" if len(r["Eligibility_Criteria"]) > 2500 else ""))
        st.link_button("Open on ClinicalTrials.gov ↗", r["Study_URL"])


# --------------------------------------------------------------------------------------
# Page 1: Find trials (patient-facing)
# --------------------------------------------------------------------------------------
if page == "Find trials":
    st.markdown("""<div class='hero'><h1>Find cancer clinical trials</h1>
    <p>Describe your cancer and the kind of treatment you are interested in. We'll find recruiting
    studies that match, grouped by research theme, so you can take a shortlist to your oncologist.</p></div>""",
                unsafe_allow_html=True)
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    all_countries = sorted({c for xs in T["Country_List"] for c in xs})
    c1, c2 = st.columns(2)
    cancer = c1.text_input("Cancer type", "non-small cell lung cancer",
                           help="e.g. breast cancer HER2 positive, prostate, melanoma, lymphoma")
    treatment = c2.text_input("Treatment interest (optional)", "immunotherapy",
                              help="e.g. immunotherapy, targeted therapy, surgery, radiotherapy, exercise")
    c3, c4, c5 = st.columns(3)
    country = c3.selectbox("Country", ["All"] + all_countries, index=0,
                           help="Countries listed on the registry record (not a guarantee a local site is open).")
    age_label = c4.selectbox("Patient age group", ["Any", "Child (under 18)", "Adult (18–64)", "Older adult (65+)"])
    age_group = {"Any": "Any", "Child (under 18)": "CHILD", "Adult (18–64)": "ADULT",
                 "Older adult (65+)": "OLDER_ADULT"}[age_label]
    sex_label = c5.selectbox("Patient sex", ["Any", "Female", "Male"])
    sex = {"Any": "Any", "Female": "FEMALE", "Male": "MALE"}[sex_label]
    c6, c7, c8 = st.columns(3)
    status_label = c6.selectbox("Trial status", ["Recruiting now", "Recruiting + opening soon", "All statuses (research view)"])
    status = {"Recruiting now": "recruiting", "Recruiting + opening soon": "recruiting_upcoming",
              "All statuses (research view)": "all"}[status_label]
    phases = c7.multiselect("Trial phase", sorted(T["Phase_Clean"].unique()), placeholder="Any phase")
    method_label = c8.selectbox("Search method", ["K-means + safety net (recommended)",
                                                  "K-means clusters only", "Search everything"])
    method = {"K-means + safety net (recommended)": "cluster_safety",
              "K-means clusters only": "cluster", "Search everything": "global"}[method_label]

    query = f"{cancer} {treatment}".strip()
    if st.button("Search trials", type="primary") or query:
        try:
            res, info = model.search(query, method=method, n_clusters=2, top_n=10, country=country,
                                     phases=phases, status=status, age_group=age_group, sex=sex)
        except ValueError as e:
            st.warning(str(e)); st.stop()

        cent = model.centroid_similarity(query)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trials shown", len(res))
        m2.metric("Best-matching theme", f"Cluster {cent.iloc[0]['Cluster']}")
        m3.metric("Trials compared", f"{info['candidates_scored']:,}", help="Trials scored by cosine similarity")
        m4.metric("Search time", f"{info['seconds'] * 1000:.0f} ms")
        if method != "global":
            st.caption(f"Your search best matches **{cent.iloc[0]['Theme']}** and **{cent.iloc[1]['Theme']}**. "
                       f"We ranked trials in those two clusters"
                       + (" and checked the rest of the catalogue for strong matches." if method == "cluster_safety" else "."))

        if res.empty:
            st.info("No trials match those filters. Try 'All' countries, 'Recruiting + opening soon', or broader words.")
        for _, r in res.iterrows():
            trial_card(r)

        st.download_button("Download this shortlist (CSV) to share with your doctor",
                           res[["Rank", "NCT_ID", "Title", "Conditions", "Interventions", "Phase_Clean",
                                "Status", "Countries", "Similarity", "Study_URL"]].to_csv(index=False),
                           file_name="trial_shortlist.csv")


# --------------------------------------------------------------------------------------
# Page 2: How the search works (K-means → cosine)
# --------------------------------------------------------------------------------------
elif page == "How the search works":
    st.title("How K-means and cosine similarity work together")
    st.markdown("""
1. **TF-IDF** turns each trial description (title, conditions, interventions, summary) into numbers.
2. **Truncated SVD** compresses those into a smaller set of features.
3. **K-means** groups the trials into **K clusters**; each has a centre (*centroid*).
4. Your search is converted the same way, then **cosine similarity with each centroid** picks the nearest research themes.
5. **Cosine similarity with each trial** inside those clusters ranks the results.
6. A **safety net** adds strong matches from other clusters, because clusters overlap.
""")
    q = st.text_input("Try a search", "HER2 positive breast cancer targeted therapy")
    try:
        cent = model.centroid_similarity(q)
    except ValueError as e:
        st.warning(str(e)); st.stop()

    st.subheader("Step 1 — Which clusters does the search match?")
    cent["Selected"] = np.where(cent.index < 2, "Searched (top 2)", "Not searched")
    chart = alt.Chart(cent).mark_bar().encode(
        x=alt.X("Centroid_Similarity:Q", title="Cosine similarity to cluster centroid"),
        y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=340, labelOverlap=False)),
        color=alt.Color("Selected:N", scale=alt.Scale(domain=["Searched (top 2)", "Not searched"],
                                                      range=["#1b7f8c", "#c9d6db"]), legend=alt.Legend(orient="bottom")),
        tooltip=["Cluster", "Theme", "Trials", alt.Tooltip("Centroid_Similarity", format=".3f")],
    ).transform_calculate(label="'Cluster ' + datum.Cluster + ' · ' + datum.Theme").properties(height=36 * len(cent))
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Step 2 — Rank trials inside the chosen clusters")
    res, info = model.search(q, method="cluster_safety", n_clusters=2, top_n=10)
    glob, ginfo = model.search(q, method="global", top_n=10)
    a, b, c = st.columns(3)
    a.metric("Trials scored (cluster-assisted)", f"{info['candidates_scored']:,}")
    b.metric("Trials scored (search everything)", f"{ginfo['candidates_scored']:,}")
    overlap = len(set(res["NCT_ID"]) & set(glob["NCT_ID"]))
    c.metric("Same top-10 as full search", f"{overlap}/10")
    st.dataframe(res[["Rank", "NCT_ID", "Title", "Cluster", "Theme", "Similarity", "Found_Via"]],
                 hide_index=True, use_container_width=True)
    st.caption("Cosine similarity measures overlap in wording, not medical suitability.")


# --------------------------------------------------------------------------------------
# Page 3: Explore research themes (clusters)
# --------------------------------------------------------------------------------------
elif page == "Explore research themes":
    st.title("Research themes discovered by K-means")
    st.caption("Theme names are generated from each cluster's most frequent terms. They are interpretations, not validated medical categories.")

    summ = (T.groupby("Cluster").agg(Trials=("NCT_ID", "size"), Recruiting=("is_recruiting", "sum"),
                                     Median_Enrolment=("Enrollment", "median"),
                                     NZ_Trials=("Country_List", lambda s: sum("New Zealand" in xs for xs in s)))
            .reset_index())
    summ["Theme"] = summ["Cluster"].map(model.cluster_labels)
    summ["Share"] = summ["Trials"] / summ["Trials"].sum()

    bar = alt.Chart(summ).mark_bar(color="#1b7f8c").encode(
        x=alt.X("Trials:Q"), y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=340, labelOverlap=False)),
        tooltip=["Cluster", "Theme", "Trials", alt.Tooltip("Share:Q", format=".0%")]
    ).transform_calculate(label="'Cluster ' + datum.Cluster + ' · ' + datum.Theme").properties(height=34 * model.k)
    st.altair_chart(bar, use_container_width=True)
    big = summ.loc[summ["Share"].idxmax()]
    if big["Share"] > 0.3:
        st.warning(f"Cluster {big['Cluster']} holds {big['Share']:.0%} of trials. A very large cluster usually means a "
                   "broad, mixed group; try a higher K in Model settings and compare.")

    st.subheader("Compare clusters")
    phase_mix = (T.groupby(["Cluster", "Phase_Clean"]).size().rename("n").reset_index())
    PHASE_ORDER = ["Early Phase 1", "Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Phase 4", "Not recorded"]
    phase_mix["phase_rank"] = phase_mix["Phase_Clean"].map({p: i for i, p in enumerate(PHASE_ORDER)})
    phase_mix["Theme"] = "Cluster " + phase_mix["Cluster"].astype(str)
    st.altair_chart(alt.Chart(phase_mix).mark_bar().encode(
        x=alt.X("n:Q", stack="normalize", title="Share of trials", axis=alt.Axis(format="%")),
        y=alt.Y("Theme:N", title="Cluster"),
        color=alt.Color("Phase_Clean:N", title="Phase", sort=PHASE_ORDER,
                        scale=alt.Scale(domain=PHASE_ORDER, range=["#cde8ea", "#9fd3d8", "#6dbcc4", "#3fa2ad",
                                                                   "#1b7f8c", "#135f6a", "#0b3f47", "#c4c9cc"])),
        order=alt.Order("phase_rank:Q"),
        tooltip=["Cluster", "Phase_Clean", "n"]).properties(height=30 * model.k), use_container_width=True)
    st.dataframe(summ[["Cluster", "Theme", "Trials", "Recruiting", "Median_Enrolment", "NZ_Trials"]]
                 .rename(columns={"Median_Enrolment": "Median planned enrolment", "NZ_Trials": "Trials listing NZ"}),
                 hide_index=True, use_container_width=True)

    st.subheader("Look inside a cluster")
    c = st.selectbox("Cluster", range(model.k), format_func=lambda x: f"Cluster {x} · {model.cluster_labels[x]}")
    st.markdown(" ".join(f"<span class='pill'>{t}</span>" for t in model.cluster_terms[c]), unsafe_allow_html=True)
    members = T[T["Cluster"] == c]
    # Representative trials = closest to the centroid
    sims = model.X_svd[members.index] @ model.centroids[c]
    rep = members.assign(Closeness=sims.round(3)).sort_values("Closeness", ascending=False)
    st.dataframe(rep[["NCT_ID", "Title", "Conditions", "Phase_Clean", "Status", "Closeness", "Study_URL"]].head(25),
                 hide_index=True, use_container_width=True,
                 column_config={"Study_URL": st.column_config.LinkColumn("Record", display_text="Open ↗"),
                                "Closeness": st.column_config.NumberColumn(help="Cosine similarity to the cluster centroid")})


# --------------------------------------------------------------------------------------
# Page 4: Data overview (CRISP-DM 2-3)
# --------------------------------------------------------------------------------------
elif page == "Data overview":
    st.title("Data understanding and preparation")
    st.caption("Preparation steps follow notebooks/latest.ipynb (cells 27–54).")
    st.dataframe(pd.DataFrame(list(prep_log.items()), columns=["Preparation milestone", "Records"]),
                 hide_index=True, use_container_width=False)
    st.markdown("**Status of interventional trials** (only *Recruiting* is searched by default)")
    stc = T["Status"].value_counts().rename_axis("Status").reset_index(name="Trials")
    st.altair_chart(alt.Chart(stc).mark_bar().encode(
        x="Trials:Q", y=alt.Y("Status:N", sort="-x", title=None),
        color=alt.condition(alt.datum.Status == "RECRUITING", alt.value("#1b7f8c"), alt.value("#c4c9cc"))),
        use_container_width=True)

    l, r = st.columns(2)
    ph = T["Phase_Clean"].value_counts().rename_axis("Phase").reset_index(name="Trials")
    l.markdown("**Trial phase**")
    l.altair_chart(alt.Chart(ph).mark_bar(color="#1b7f8c").encode(x="Trials:Q", y=alt.Y("Phase:N", sort="-x", title=None)),
                   use_container_width=True)
    missing_phase = (T["Phase_Clean"] == "Not recorded").mean()
    l.caption(f"{missing_phase:.0%} have no phase recorded; shown as 'Not recorded' rather than guessed.")

    ct = T["Country_List"].explode().value_counts().head(15).rename_axis("Country").reset_index(name="Trials")
    r.markdown("**Top listed countries** (one trial can list many)")
    r.altair_chart(alt.Chart(ct).mark_bar(color="#0f4c5c").encode(x="Trials:Q", y=alt.Y("Country:N", sort="-x", title=None)),
                   use_container_width=True)

    en = T["Enrollment"].dropna()
    bands = ["1–20", "21–50", "51–100", "101–250", "251–500", "501–1,000", "1,000+"]
    binned = pd.cut(en, [0, 20, 50, 100, 250, 500, 1000, np.inf], labels=bands).value_counts().reindex(bands)
    st.markdown(f"**Planned enrolment per trial** (median {en.median():,.0f} participants)")
    st.altair_chart(alt.Chart(binned.rename_axis("Band").reset_index(name="Trials")).mark_bar(color="#1b7f8c").encode(
        x=alt.X("Band:N", sort=bands, title="Planned participants", axis=alt.Axis(labelAngle=0)), y="Trials:Q"),
        use_container_width=True)
    st.caption("Planned enrolment is the target set at registration, not actual recruitment.")

    raw_cols = {"Study Title": "Title", "Conditions": "Conditions", "Brief Summary": "Summary",
                "Interventions": "Interventions", "Countries": "Countries", "Eligibility Criteria": "Eligibility_Criteria",
                "Phases": "Phase_Clean"}
    miss = pd.DataFrame({"Field": list(raw_cols), "Missing %": [
        ((T[c] == "") | (T[c] == "Not recorded")).mean() * 100 for c in raw_cols.values()]})
    st.markdown("**Missing values in model-ready trials**")
    st.dataframe(miss.round(1), hide_index=True)


# --------------------------------------------------------------------------------------
# Page 5: Model evaluation (CRISP-DM 5)
# --------------------------------------------------------------------------------------
elif page == "Model evaluation":
    st.title("Model evaluation")
    n_tr = int((T["Split"] == "Train").sum())
    st.markdown(f"Current model: **K = {model.k}**, **{model.svd_components} SVD components** "
                f"(explaining {model.explained_variance:.0%} of TF-IDF variance). TF-IDF, SVD and K-means were fitted on "
                f"the **80% training split ({n_tr:,} trials)**, the same `train_test_split(random_state=42)` as the notebook; "
                f"the other {len(T) - n_tr:,} test trials are assigned to the nearest centroid.")

    @st.cache_data(show_spinner="Testing K = 3…12")
    def ksel(_m, key):
        return k_selection(_m.X_svd, _m.trials["Split"].to_numpy())

    @st.cache_data(show_spinner="Checking stability across random seeds")
    def stab(_m, key):
        return stability(_m.X_svd, _m.k)

    @st.cache_data(show_spinner="Comparing search methods")
    def comp(_m, key):
        return compare_methods(_m)

    key = (model.k, model.svd_components, len(T))
    st.subheader("1 · Choosing K: elbow and silhouette")
    ks = ksel(model, key)
    l, r = st.columns(2)
    l.altair_chart(alt.Chart(ks).mark_line(point=True, color="#0f4c5c").encode(x=alt.X("K:O", axis=alt.Axis(labelAngle=0)), y=alt.Y("Inertia:Q", scale=alt.Scale(zero=False)))
                   .properties(title="Elbow method"), use_container_width=True)
    sil_long = ks.melt(id_vars="K", value_vars=["Train silhouette", "Test silhouette"], var_name="Set", value_name="Silhouette")
    r.altair_chart(alt.Chart(sil_long).mark_line(point=True).encode(
        x=alt.X("K:O", axis=alt.Axis(labelAngle=0)), y="Silhouette:Q",
        color=alt.Color("Set:N", scale=alt.Scale(range=["#1b7f8c", "#f0a202"]), legend=alt.Legend(orient="bottom")))
        .properties(title="Cosine silhouette: train vs unseen test"), use_container_width=True)
    ks["Silhouette"] = ks["Test silhouette"]
    best = ks.loc[ks["Silhouette"].idxmax()]
    cur = ks.loc[ks["K"] == model.k, "Silhouette"]
    cur_txt = f"; current K = {model.k} scores {cur.iloc[0]:.3f}" if len(cur) else ""
    st.caption(f"Test silhouette tracks train closely, so the clusters generalise to unseen trials. Highest test silhouette at K = {int(best['K'])} ({best['Silhouette']:.3f}){cur_txt}. All scores are well "
               "below 0.5, so clusters overlap. That is common for medical text, which is why the safety net exists "
               "and why interpretability is judged alongside the numbers.")

    st.subheader("2 · Stability")
    s = stab(model, key)
    st.metric("Mean Adjusted Rand Index across 5 random seeds", f"{s['Adjusted_Rand_Index'].mean():.2f}",
              help="1 = identical clusters every run; 0 = no better than chance")

    st.subheader("3 · Does K-means help the search?")
    st.markdown("Using the recruiting catalogue, for 10 patient-style queries we compare the cluster-assisted search with searching every trial. "
                "**Overlap** shows how many of the full-search top 10 are still found (a check for missed trials); "
                "**reduction** shows how much smaller the search space becomes.")
    cm = comp(model, key)
    agg = cm.groupby("Method").agg(Overlap=("Overlap_with_global_top10", "mean"),
                                   Reduction=("Search_space_reduction", "mean")).reset_index()
    agg["Method"] = agg["Method"].map({"cluster": "K-means clusters only", "cluster_safety": "K-means + safety net"})
    x1, x2 = st.columns(2)
    for col, (_, row) in zip((x1, x2), agg.iterrows()):
        col.metric(row["Method"], f"{row['Overlap']:.0%} overlap", f"{row['Reduction']:.0%} fewer trials scored",
                   delta_color="off")
    pivot = cm.pivot(index=["Query_ID", "Query"], columns="Method", values="Overlap_with_global_top10").reset_index()
    st.dataframe(pivot.rename(columns={"cluster": "Clusters only", "cluster_safety": "With safety net"}),
                 hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.ProgressColumn(c, format="%.0f%%", min_value=0, max_value=1)
                                for c in ["Clusters only", "With safety net"]})
    st.info("Overlap is agreement with full search, not accuracy. To report Precision@5, download the review sheet, "
            "have group members label each result 0/1 without seeing the method, and compute the mean.")

    rows = []
    for qid, q in EVALUATION_QUERIES.items():
        for m in ("global", "cluster_safety"):
            r_, _ = model.search(q, m, top_n=5)
            for _, x in r_.iterrows():
                rows.append({"Query_ID": qid, "Query": q, "Method": m, "Rank": x["Rank"], "NCT_ID": x["NCT_ID"],
                             "Title": x["Title"], "Conditions": x["Conditions"], "Relevance_0_or_1": ""})
    rev = pd.DataFrame(rows).sample(frac=1, random_state=1)  # shuffled so reviewers can't infer method order
    st.download_button("Download blinded relevance-review sheet (CSV)", rev.to_csv(index=False),
                       file_name="relevance_review_template.csv")
