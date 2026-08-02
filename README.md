# Board Game Data Analytics 🎲📊

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20Pandas-orange)](#)
[![Status](https://img.shields.io/badge/Status-Expanding_with_IR-green)](#)

> **Original Data Mining project (notebooks 0-5):** Tommaso Agostini, Elisa Calabrese
> *Developed for the Data Mining course at the University of Pisa (A.Y. 2025/2026).*
>
> **Information Retrieval extension (notebooks 6-8):** Tommaso Agostini (solo)
> *Designed, implemented, and documented independently as an addition to the original coursework, for my personal portfolio - not part of the original group submission.*
>
> 📄 **[Read the full Original Project Report (PDF)](ProjectReport_DM1.pdf)**

## 📌 Project Overview
This project involves a comprehensive data mining analysis of a board game dataset. The study covers the entire data mining pipeline, from data preparation and feature engineering to the application of unsupervised and supervised machine learning techniques. 

The original dataset comprised 21,926 board games rated by an online community. After rigorous cleaning (handling outliers, logic inconsistencies, and imputing missing values via a category-aware median strategy), the final dataset was refined to 19,728 records and 25 highly informative attributes.

## 🚀 Key Findings & Models

### 1. Clustering (Discovering Communities)
We explored the structural relationships using K-Means, Hierarchical Clustering, DBSCAN, and OPTICS. The data exhibited a continuous density gradient (a "continent") rather than isolated islands.
* **Winner:** **OPTICS** successfully identified density "valleys" across the continuous distribution without over-generalizing, outperforming standard DBSCAN.

### 2. Classification (Predicting Ratings & Playtime)
We tested K-Nearest Neighbors (K-NN), Categorical Naive Bayes, and Decision Trees.
* **Binary Task (Long vs. Short Playtime):** K-NN emerged as the top performer (Accuracy 81%, AUC 0.89).
* **Multiclass Task (Low, Medium, High Rating):** K-NN again achieved the best F1-score (67%), successfully handling the heavy overlap between contiguous rating categories better than tree-based models.

### 3. Regression (Estimating Complexity & Popularity)
We predicted game complexity (`GameWeight`) and popularity using Linear, Ridge, Lasso, Decision Tree, and K-NN Regressors.
* **Winner:** Non-linear models outperformed linear approaches. For `GameWeight`, **K-NN** achieved the highest R-squared (0.568), proving that interactions between playtime, expansions, and complexity are strictly non-linear.

### 4. Pattern Mining (Association Rules)
Using the Apriori algorithm (min support 5%, min confidence 70%), we extracted frequent itemsets that characterized niche dependencies.
* **Insight:** High-lift rules (> 5.0) accurately characterized the "Wargames" genre as being tightly associated with very long playtimes, high complexity, and low player counts. Used as a manual classifier, this rule alone achieved a 75% precision.

### 5. Information Retrieval (Searching by Natural Language)
The original Data Mining pipeline dropped the free-text `Description` field to focus on numerical/categorical features. This extension recovers it and builds a small search engine on top, comparing three retrieval strategies of increasing sophistication:

* **Lexical baseline (BM25):** an inverted index over the tokenized descriptions (`rank_bm25`), used both as a standalone baseline and as one half of the candidate pool for later stages.
* **Dense retrieval (FAISS):** descriptions are embedded with `all-MiniLM-L6-v2` (`sentence-transformers`) and indexed with an approximate k-NN HNSW index (`faiss`), enabling semantic matches BM25 misses (e.g. "space betrayal game" retrieving thematically relevant titles with little keyword overlap).
* **Learning-to-Rank re-ranking:** candidates from BM25 + FAISS are pooled and re-ranked with a **Ridge regression** trained on relevance labels, using BM25 score, semantic similarity, community rating, and popularity as features.
* **Evaluation ground truth:** since no real user click data exists for this dataset, relevance judgments (qrels) for 25 test queries were generated with an **LLM-as-a-Judge** approach (`gemini-3.5-flash-lite`) rather than by hand. The LLM ranks pooled candidates per query, and 0-3 relevance labels are derived from fixed rank buckets (rather than asked for directly) to keep label distributions comparable across queries. Metrics (MRR, NDCG@5) are reported on a held-out 20% split of these queries.
* **Known limitation:** 25 LLM-generated qrels is a small, single-judge ground truth - useful for demonstrating the evaluation methodology end-to-end, but not a substitute for human-annotated or real-interaction-based relevance data at production scale.

---

## 📂 Repository Structure
* **`data/`**: Raw dataset, cleaned versions, and generated association rules.
  * Generated at runtime and **not** committed (see `.gitignore`): `embeddings.npy`, `faiss_index.bin`, `tokens.pkl` - these are deterministic caches, rebuilt automatically on first run of the relevant notebook.
  * **`data/evaluation/`**: `automated_qrels.json` (the LLM-generated evaluation ground truth) **is** committed, since it's small and is itself a project artifact worth reviewing. `automated_qrels_raw.json` (the cached raw LLM responses, used only for debugging/recovery) is **not** committed.
* **`notebooks/`**:
  * `0_data_understanding.ipynb`: EDA and feature correlations.
  * `1_data_preparation.ipynb`: Missing values imputation, encoding, and scaling (log1p).
  * `2_clustering.ipynb`: Unsupervised functional partitions.
  * `3_classification.ipynb`: Binary and Multiclass predictions.
  * `4_regression.ipynb`: Continuous target estimation.
  * `5_pattern_mining.ipynb`: Apriori and frequent itemsets.
  * `6_information_retrieval_baseline.ipynb`: BM25 lexical search baseline.
  * `7_neural_ir_faiss.ipynb`: dense retrieval with sentence embeddings + FAISS.
  * `8_learning_to_rank.ipynb`: candidate pooling, LLM-as-a-Judge qrel generation, and the Ridge re-ranker, with the final BM25-vs-LTR evaluation.
* **`scripts/generate_qrels.py`**: standalone batch script that calls the Gemini API to generate `data/evaluation/automated_qrels.json`. Run separately from the notebooks (see below) so the API key never needs to touch a notebook cell.

---

## 🛠️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/ecalabrese17/Board-Game-Data-Analytics.git
   cd Board-Game-Data-Analytics
   ```

2. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Place the raw dataset files in `data/` (`cleaned_dataset.csv`, `DM1_game_dataset.csv`) - see `data/README.md` for the source if you don't already have them.

4. Run the notebooks 0 through 5 in order for the original Data Mining pipeline (clustering, classification, regression, pattern mining).

### Running the Information Retrieval extension (notebooks 6-8)

5. Set your Gemini API key as an environment variable (used only by `scripts/generate_qrels.py`, never hardcoded in any notebook):
   ```bash
   export GEMINI_API_KEY="your-key-here"  # Windows: set GEMINI_API_KEY=your-key-here
   ```
   Get a free-tier key at [Google AI Studio](https://aistudio.google.com/apikey). The script is written to work with the free tier's rate limits (built-in throttling, retries, and checkpointing - safe to interrupt and resume).

6. Run `6_information_retrieval_baseline.ipynb` and `7_neural_ir_faiss.ipynb` (in either order - they share the same embeddings/FAISS cache, so encoding the ~20k descriptions only happens once, in whichever you run first).

7. Generate the evaluation ground truth (skips this step and reuses `data/evaluation/automated_qrels.json` if it's already present):
   ```bash
   python scripts/generate_qrels.py
   ```

8. Run `8_learning_to_rank.ipynb` for the candidate pooling, the trained Ridge re-ranker, and the final BM25-vs-LTR comparison on the held-out test queries.