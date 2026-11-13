# Cancer Trial Finder: Streamlit prototype
BSNS8030 · SDG 3 Good Health and Well-being · CRISP-DM Stage 6 (Deployment)

A patient-facing tool for discovering recruiting cancer clinical trials. It combines
**K-means clustering** (to group trials into research themes) with **cosine similarity**
(to rank individual trials against a patient's search).

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501.

## Data
`data/cancer_trials_10000.csv` is the group's 10,000-record ClinicalTrials.gov export from
`notebooks/latest.ipynb`, using the notebook's column names (NCT Number, Study Title, Study Status,
Brief Summary, Conditions, Interventions, Phases, Countries, Eligibility Criteria and so on).

## Pipeline (model.py)
1. **Preparation:** mirrors `latest.ipynb` cells 27–54. It normalises whitespace, keeps valid NCT IDs,
   deduplicates, keeps interventional trials only (7,921), cleans the text (tumour → tumor), builds
   `weighted_model_text` (conditions ×2, interventions ×2, title, summary), drops trials with under 25
   characters of text, and flags recruiting (1,188) and not-yet-recruiting (402) trials.
2. **Train/test split:** `train_test_split(test_size=0.20, random_state=42)`, the same as the notebook.
   TF-IDF, SVD and K-means are fitted on the 6,336 training trials only; the 1,585 test trials are
   assigned to their nearest centroid.
3. **TF-IDF:** unigrams and bigrams, 5,000 features. Stop-words include generic trial words, the API's
   intervention-type prefixes (DRUG:, PROCEDURE:, ...) and a phrase the National Cancer Institute
   repeats in many summaries. Without these, clusters formed around intervention type, not cancer type.
4. **Truncated SVD** (80 components) → **K-means** (K = 8 by default).
5. **Search:** runs over the **recruiting catalogue** by default. Query → TF-IDF → SVD → cosine
   similarity to each centroid → top 2 clusters → cosine similarity to each trial → ranking, plus a
   safety net for strong matches in other clusters. Patients can filter by country, age group, sex,
   phase and status.

## Results on this data (K = 8)
- Themes: Carcinoma/Chemotherapy, Lung, Solid tumours, Leukaemia/Lymphoma/Myeloma, Prostate,
  Surgery/Imaging, Supportive care (exercise, quality of life) and Breast/HER2.
- Cosine silhouette: 0.132 on train and 0.135 on test. Test closely matches train, so the clusters
  generalise; the low values mean the clusters overlap.
- Stability across seeds: mean Adjusted Rand Index of 0.79.
- Search: cluster-assisted search scores about 70% fewer trials. It finds 91% of the full-search
  top 10 using clusters only, and 100% with the safety net.

## App pages
| Page | CRISP-DM | What it shows |
|---|---|---|
| Find trials | Deployment | Patient search with country and phase filters; shortlist download |
| How the search works | Modelling | Query-to-centroid similarities, then the ranking inside clusters |
| Explore research themes | Modelling / Evaluation | Cluster sizes, themes, phase mix, enrolment, NZ trials, representative trials |
| Data overview | Data understanding / preparation | Record counts, phases, countries, enrolment, missingness |
| Model evaluation | Evaluation | Elbow, silhouette, stability (ARI), cluster vs global search, blinded Precision@5 review sheet |

## Limitations (say these in the presentation)
- Ranks trials by how similar their **text** is to the search. It does **not** check eligibility and is not medical advice.
- This is a snapshot; recruitment status changes, so always check the official record.
- Silhouette scores are low (about 0.13 at K = 8), so clusters overlap. That is why the safety net exists.
- Phase is missing for about 42% of the raw export (mostly observational studies) and about 27% of interventional trials. These are shown as 'Not recorded', not guessed.
- The "overlap with global search" metric is agreement, not accuracy. Precision@5 needs human relevance labels.
