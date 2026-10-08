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
The original Data Mining pipeline dropped the free-text `Description` field to focus on numerical and categorical features. This extension restores the natural-language descriptions and builds a retrieval pipeline that progresses from lexical retrieval to dense retrieval, heuristic hybrid retrieval, and finally Learning-to-Rank.

The natural description (`Description_nat`) is the common textual source for both BM25 and dense retrieval, keeping the retrieval stages methodologically consistent.

* **BM25 baseline (NB6):** tokenized `Description_nat` documents are indexed with `rank_bm25`. The resulting `tokens.pkl` cache is reused by later stages.
* **Dense retrieval (NB7):** `Description_nat` is encoded with `all-MiniLM-L6-v2` (`sentence-transformers`) and stored in `embeddings_nat.npy`. An approximate FAISS HNSW index is built for semantic retrieval.
* **LLM-as-a-Judge qrels:** because no real user click data exists for this dataset, relevance judgments are generated for 100 natural-language queries. For each query, the candidate pool is the union of BM25 top-20 and dense top-20 results. Gemini ranks the candidates by `BGGId`; fixed rank buckets are then converted into relevance labels (top 2 → 3, next 6 → 2, next 8 → 1, remaining → 0). The resulting `expanded_qrels_v4.json` is used as the frozen evaluation ground truth.
* **Heuristic hybrid retrieval (NB8):** BM25 and dense retrieval are combined through a heuristic reranking stage and evaluated on the full 19,728-game corpus. This is an intermediate hybrid-retrieval baseline, not Learning-to-Rank.
* **Learning-to-Rank (NB9):** the qrels-v4 candidate pool is reranked with a pointwise Ridge model using BM25 score, semantic distance, normalized rating, and normalized popularity. The model is trained and evaluated with query-grouped 5-fold cross-validation, avoiding query leakage between train and test folds.

The evaluation uses MRR and NDCG@5. The two final stages have different evaluation scopes: NB8 measures retrieval over the full corpus, while NB9 measures reranking quality inside the frozen judged candidate pool. Their absolute metrics should therefore not be interpreted as a direct apples-to-apples comparison.

**NB8 full-corpus evaluation:**
* BM25: **MRR 0.817**, **NDCG@5 0.397**
* Heuristic Hybrid: **MRR 0.827**, **NDCG@5 0.374**

**NB9 5-fold query-grouped evaluation:**
* BM25 candidate-pool baseline: **MRR 0.822**, **NDCG@5 0.399**
* Ridge LTR: **MRR 0.894**, **NDCG@5 0.502**
* Improvement over the candidate-pool BM25 baseline: **+0.072 MRR**, **+0.103 NDCG@5**

**Known limitations:** the qrels are LLM-generated and therefore represent a proxy for human relevance judgments. The evaluation also uses a fixed candidate-generation strategy and a relatively small judged set compared with the full corpus. The qrels generator is resumable and uses deterministic model settings, retries, and coverage checks, but it should not be treated as a substitute for human annotation or real interaction data.

---

## 📂 Repository Structure
* **`data/`**: Raw dataset, cleaned versions, association rules, and generated retrieval artifacts.
  * Runtime caches are **not** committed (see `.gitignore`): `tokens.pkl`, `embeddings_nat.npy`, and `faiss_index.bin`.
  * **`data/expanded_qrels_v4.json`** (or the repository's configured evaluation path): the frozen LLM-generated qrels used by NB8/NB9.
  * Raw/checkpoint LLM responses are not required for normal evaluation and should not be committed.
* **`notebooks/`**:
  * `0_data_understanding.ipynb`: EDA and feature correlations.
  * `1_data_preparation.ipynb`: Missing-value imputation, encoding, and scaling.
  * `2_clustering.ipynb`: Unsupervised clustering.
  * `3_classification.ipynb`: Binary and multiclass prediction.
  * `4_regression.ipynb`: Continuous target estimation.
  * `5_pattern_mining.ipynb`: Apriori and frequent itemsets.
  * `6_information_retrieval_baseline.ipynb`: BM25 baseline over `Description_nat`; creates the token cache.
  * `7_neural_ir_faiss.ipynb`: dense retrieval over `Description_nat`; creates the embedding cache and FAISS index.
  * `8_hybrid_retrieval_and_evaluation.ipynb`: full-corpus BM25 retrieval, heuristic hybrid reranking, and evaluation.
  * `9_learning_to_rank.ipynb`: actual pointwise Ridge Learning-to-Rank with query-grouped cross-validation and ablation analysis.
* **`scripts/generate_qrels_v4.py`**: standalone Gemini batch script that builds the frozen qrels from the BM25 top-20 ∪ dense top-20 candidate pools. It consumes the NB6/NB7 artifacts and does not regenerate embeddings.

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
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Place the raw dataset files in `data/` (`cleaned_dataset.csv`, `DM1_game_dataset.csv`) and follow `data/README.md` for the data source.

4. Run notebooks 0 through 5 in order for the original Data Mining pipeline.

### Running the Information Retrieval extension (notebooks 6-9)

5. Set the Gemini API key as an environment variable. It is used only by the qrels-generation script:
   ```bash
   export GEMINI_API_KEY="your-key-here"
   ```
   On Windows:
   ```powershell
   set GEMINI_API_KEY=your-key-here
   ```

6. Run NB6 and NB7 to create the shared retrieval artifacts:
   ```text
   NB6 → data/tokens.pkl
   NB7 → data/embeddings_nat.npy + data/faiss_index.bin
   ```
   The notebooks reuse existing caches when available.

7. Generate the frozen qrels:
   ```bash
   python scripts/generate_qrels_v4.py
   ```
   The script uses the BM25 top-20 ∪ dense top-20 pool, Gemini ranking, fixed relevance buckets, retries, and checkpointing. Re-running it resumes from `expanded_qrels_v4.json`.

8. Run NB8 for full-corpus retrieval evaluation:
   ```text
   NB8 → BM25 vs Heuristic Hybrid
   ```

9. Run NB9 for actual Learning-to-Rank:
   ```text
   NB9 → Ridge LTR + 5-fold query-grouped CV + ablations
   ```

The intended dependency chain is:

```text
cleaned_dataset.csv + DM1_game_dataset.csv + details.csv
                         │
                         ▼
              Description_nat
                 ┌───────┴───────┐
                 ▼               ▼
                NB6             NB7
                 │               │
          tokens.pkl       embeddings_nat.npy
                                 │
                                 ▼
                         FAISS HNSW index
                 └───────┬───────┘
                         ▼
                generate_qrels_v4.py
                         │
                         ▼
                  expanded_qrels_v4
                    ┌────┴────┐
                    ▼         ▼
                   NB8       NB9
                retrieval    LTR
                evaluation  reranking
```

