# Board Game Data Analytics 🎲📊

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20Pandas-orange)](#)
[![Status](https://img.shields.io/badge/Status-Expanding_with_IR-green)](#)

> **Authors:** Tommaso Agostini, Elisa Calabrese
> *This project was originally developed for the Data Mining course at the University of Pisa (A.Y. 2025/2026). I have subsequently refactored and enhanced the codebase for my personal portfolio.*
> 
> 📄 **[Read the full Original Project Report (PDF)](ProjectReport_DM1.pdf)**

## 📌 Project Overview
This project involves a comprehensive data mining analysis of a board game dataset. The study covers the entire data mining pipeline, from data preparation and feature engineering to the application of unsupervised and supervised machine learning techniques. 

The original dataset comprised 21,926 board games rated by an online community. After rigorous cleaning (handling outliers, logic inconsistencies, and imputing missing values via a category-aware median strategy), the final dataset was refined to 19,729 records and 25 highly informative attributes.

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

---

## 📂 Repository Structure
* **`data/`**: Raw dataset, cleaned versions, and generated association rules.
* **`notebooks/`**:
  * `0_data_understanding.ipynb`: EDA and feature correlations.
  * `1_data_preparation.ipynb`: Missing values imputation, encoding, and scaling (log1p).
  * `2_clustering.ipynb`: Unsupervised functional partitions.
  * `3_classification.ipynb`: Binary and Multiclass predictions.
  * `4_regression.ipynb`: Continuous target estimation.
  * `5_pattern_mining.ipynb`: Apriori and frequent itemsets.

---

## 🛠️ How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/ecalabrese17/Board-Game-Data-Analytics.git](https://github.com/ecalabrese17/Board-Game-Data-Analytics.git)
   cd Board-Game-Data-Analytics