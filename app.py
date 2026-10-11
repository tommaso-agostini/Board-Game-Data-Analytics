import os
import pickle
import string
import html
import re

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


st.set_page_config(layout="wide", page_title="Board Game Search Engine")

st.title("Board Game Search Engine")
query = st.text_input(
    label="What kind of game are you looking for?",
    placeholder="Space exploration and empire building",
)

DATA_DIR = "./data"
DETAILS_PATH = os.path.join(DATA_DIR, "details.csv")
CLEAN_DATASET_PATH = os.path.join(DATA_DIR, "cleaned_dataset.csv")
RAW_DATASET_PATH = os.path.join(DATA_DIR, "DM1_game_dataset.csv")

EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings_nat.npy")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
TOKENS_PATH = os.path.join(DATA_DIR, "tokens.pkl")
LTR_MODEL_PATH = os.path.join(DATA_DIR, "ltr_ridge_model.pkl")


# -----------------------------------------------------------------------------
# Natural-language description: same construction used by NB6/NB7/qrels.
# -----------------------------------------------------------------------------
def clean_html(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(html.unescape(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_transformer_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def load_dfs():
    # Canonical corpus construction used by the final IR pipeline.
    df_clean = pd.read_csv(CLEAN_DATASET_PATH)
    df_raw = pd.read_csv(RAW_DATASET_PATH)

    df = pd.merge(
        df_clean,
        df_raw[["BGGId", "Description"]],
        on="BGGId",
        how="inner",
    )
    df = df.dropna(subset=["Description"]).reset_index(drop=True)

    # details.csv uses `id` and lowercase `description`.
    details = pd.read_csv(DETAILS_PATH).drop_duplicates("id")
    details_map = df[["BGGId"]].merge(
        details,
        left_on="BGGId",
        right_on="id",
        how="left",
    )

    if len(details_map) != len(df):
        raise ValueError(
            "Natural-description merge changed corpus size; "
            "dataset alignment cannot be guaranteed."
        )

    natural_description = details_map["description"].map(clean_html)
    has_original = natural_description.str.len() > 0

    df["Description_nat"] = natural_description.where(
        has_original,
        df["Description"],
    )

    df = df.reset_index(drop=True)
    return df


df = load_dfs()
model = load_transformer_model()


# -----------------------------------------------------------------------------
# Required precomputed artifacts.
# -----------------------------------------------------------------------------
def require_artifact(path, label):
    if not os.path.exists(path):
        st.error(
            f"{label} not found at {path}. "
            "Run the corresponding IR pipeline notebook first."
        )
        st.stop()


require_artifact(EMBEDDINGS_PATH, "Natural-language embeddings")
require_artifact(FAISS_INDEX_PATH, "FAISS index")
require_artifact(TOKENS_PATH, "BM25 token cache")
require_artifact(LTR_MODEL_PATH, "LTR model")


embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

if embeddings.shape[0] != len(df):
    st.error(
        "embeddings_nat.npy is not aligned with the current dataset "
        f"({embeddings.shape[0]} embeddings for {len(df)} games)."
    )
    st.stop()


@st.cache_resource
def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)

    if index.ntotal != len(df):
        raise ValueError(
            f"FAISS index contains {index.ntotal} vectors, "
            f"but the dataset contains {len(df)} games."
        )

    return index


index = load_faiss_index()


with open(TOKENS_PATH, "rb") as f:
    tokens_list = pickle.load(f)

if len(tokens_list) != len(df):
    st.error(
        "tokens.pkl is not aligned with the current dataset "
        f"({len(tokens_list)} tokenized documents for {len(df)} games)."
    )
    st.stop()


df["tokens"] = tokens_list

stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words]


@st.cache_resource
def build_bm25_index(_tokens_list):
    return BM25Okapi(_tokens_list)


bm25 = build_bm25_index(tokens_list)


# -----------------------------------------------------------------------------
# NB9 LTR artifact.
# The pickle contains a dictionary with the Ridge estimator and metadata:
#   model, features, alpha, training_queries, qrels_version
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ltr_artifact():
    if not os.path.exists(LTR_MODEL_PATH):
        raise FileNotFoundError(f"Missing LTR model: {LTR_MODEL_PATH}")

    with open(LTR_MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    if not isinstance(artifact, dict):
        raise ValueError("Invalid LTR model artifact: expected a dictionary.")

    required_keys = {"model", "features"}
    missing = required_keys - set(artifact.keys())
    if missing:
        raise ValueError(
            f"Invalid LTR model artifact: missing keys {sorted(missing)}."
        )

    return artifact


ltr_artifact = load_ltr_artifact()
ltr_model = ltr_artifact["model"]
ltr_features = list(ltr_artifact["features"])


# Fail early if the saved estimator and metadata disagree.
model_feature_names = list(getattr(ltr_model, "feature_names_in_", []))
if model_feature_names and model_feature_names != ltr_features:
    raise ValueError(
        "LTR artifact feature metadata does not match the Ridge estimator: "
        f"metadata={ltr_features}, estimator={model_feature_names}"
    )


# -----------------------------------------------------------------------------
# Feature construction.
# NB9 uses the L2 geometry of FAISS for semantic_score: smaller distance is
# better, therefore the ranking feature is the negative L2 distance.
# -----------------------------------------------------------------------------
def semantic_feature_from_l2(distances):
    return -np.asarray(distances, dtype=np.float32)


def add_ltr_features(candidates_df):
    candidates_df = candidates_df.copy()

    # BM25 and semantic_score are the exact feature names expected by NB9.
    if "bm25_score" not in candidates_df.columns:
        raise ValueError("Missing required LTR feature: bm25_score")
    if "semantic_score" not in candidates_df.columns:
        raise ValueError("Missing required LTR feature: semantic_score")

    rating_col = "Rating"
    popularity_col = "Avg_popularity_log"

    if rating_col not in candidates_df.columns:
        raise ValueError(f"Missing required dataset column: {rating_col}")
    if popularity_col not in candidates_df.columns:
        raise ValueError(f"Missing required dataset column: {popularity_col}")

    # NB9 normalizes the auxiliary numerical features before fitting Ridge.
    # MinMaxScaler is fit column-wise, so using it jointly preserves the same
    # per-feature transformation without introducing cross-feature scaling.
    scaler = MinMaxScaler()
    candidates_df[["norm_rating", "norm_popularity"]] = scaler.fit_transform(
        candidates_df[[rating_col, popularity_col]]
    )

    # The saved NB9 artifact determines the exact feature order.
    missing_features = [
        feature for feature in ltr_features if feature not in candidates_df.columns
    ]
    if missing_features:
        raise ValueError(
            "The live candidate table is missing LTR features: "
            f"{missing_features}. Expected: {ltr_features}"
        )

    return candidates_df


# -----------------------------------------------------------------------------
# Retrieval + LTR reranking.
# -----------------------------------------------------------------------------
def search_games_ltr(query, top_k=10, candidate_pool_size=50):
    if not query.strip():
        return pd.DataFrame()

    # 1. Dense retrieval over Description_nat embeddings.
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(
        query_vector,
        k=min(candidate_pool_size, len(df)),
    )

    semantic_scores_dict = {
        int(df.iloc[idx]["BGGId"]): float(
            semantic_feature_from_l2([dist])[0]
        )
        for idx, dist in zip(indices[0], distances[0])
        if idx != -1
    }

    # 2. BM25 retrieval over the NB6 token cache.
    query_tokens = preprocess_text(query)
    bm25_scores = bm25.get_scores(query_tokens)

    df_temp = df.copy()
    df_temp["bm25_score"] = bm25_scores
    df_temp["semantic_score"] = (
        df_temp["BGGId"].map(semantic_scores_dict).fillna(0.0)
    )

    # 3. Candidate pool = top BM25 union top dense results.
    top_bm25_ids = set(
        df_temp.nlargest(candidate_pool_size, "bm25_score")["BGGId"]
    )
    top_semantic_ids = set(semantic_scores_dict.keys())
    candidate_ids = top_bm25_ids.union(top_semantic_ids)

    candidates_df = df_temp[df_temp["BGGId"].isin(candidate_ids)].copy()

    # 4. Construct exactly the feature schema expected by NB9.
    candidates_df = add_ltr_features(candidates_df)
    X_pred = candidates_df[ltr_features]

    candidates_df["ltr_final_score"] = ltr_model.predict(X_pred)

    return candidates_df.sort_values(
        by="ltr_final_score",
        ascending=False,
    ).head(top_k)


# -----------------------------------------------------------------------------
# UI.
# -----------------------------------------------------------------------------
if query:
    st.markdown("---")
    st.subheader(f"Search Results for: '{query}'")

    ltr_results = search_games_ltr(query, top_k=10)

    if not ltr_results.empty:
        for _, row in ltr_results.iterrows():
            with st.container():
                col_info, col_score = st.columns([4, 1])

                with col_info:
                    st.markdown(f"### {row['Name']}")

                    min_p = (
                        int(row["MinPlayers"])
                        if "MinPlayers" in row and not pd.isna(row["MinPlayers"])
                        else "N/A"
                    )
                    max_p = (
                        int(row["MaxPlayers"])
                        if "MaxPlayers" in row and not pd.isna(row["MaxPlayers"])
                        else "N/A"
                    )

                    st.caption(
                        f"Players: {min_p} - {max_p} | "
                        f"BGG ID: {int(row['BGGId'])}"
                    )

                    bm25_v = row.get("bm25_score", 0.0)
                    sem_v = row.get("semantic_score", 0.0)
                    st.text(
                        f"BM25: {bm25_v:.2f} | "
                        f"Semantic L2 feature: {sem_v:.2f}"
                    )

                with col_score:
                    st.metric(
                        label="LTR Score",
                        value=f"{row['ltr_final_score']:.4f}",
                    )

                    if "Rating" in row and not pd.isna(row["Rating"]):
                        st.caption(f"Rating: {row['Rating']:.2f}")

                st.divider()
    else:
        st.warning("No games found for this query.")
