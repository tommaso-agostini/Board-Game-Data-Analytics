"""
generate_qrels.py
Batch job to generate LLM-as-a-Judge relevance scores for board game search evaluation.
"""

import os
from dotenv import load_dotenv
import json
import time
import re
import datetime
import string
import numpy as np
import pandas as pd
import faiss
import nltk
from difflib import get_close_matches
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import google.generativeai as genai

load_dotenv()

# ==========================================
# 0. ENVIRONMENT & NLTK SETUP
# ==========================================
# Download required NLTK data quietly if not already present
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# It is highly recommended to set your API key as an environment variable in your terminal:
# export GEMINI_API_KEY="your_actual_key"
API_KEY = os.environ.get("GEMINI_API_KEY", "INSERT_YOUR_API_KEY_HERE")

# ==========================================
# CONFIGURATION
# ==========================================
DRY_RUN = False
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 20
MIN_SECONDS_BETWEEN_CALLS = 4.5
METHOD_NAME = "ranking_bucket"

N_SCORE_3 = 2   # top 2 games -> "Perfect Match"
N_SCORE_2 = 4   # next 4 games -> "Highly Relevant"
N_SCORE_1 = 6   # next 6 games -> "Partially Relevant"

qrels_file = '../data/evaluation/automated_qrels.json'
raw_responses_file = '../data/evaluation/automated_qrels_raw.json'

# ==========================================
# 1. LOAD DATA & INITIALIZE ENGINES
# ==========================================
print("Loading datasets...")
try:
    df_clean = pd.read_csv('../data/cleaned_dataset.csv')
    df_raw = pd.read_csv('../data/DM1_game_dataset.csv')
except FileNotFoundError:
    print("Error: Could not find the CSV files. Check your relative paths.")
    exit(1)

df_text = df_raw[['BGGId', 'Description']]
df = pd.merge(df_clean, df_text, on='BGGId', how='inner')
df = df.dropna(subset=['Description']).reset_index(drop=True)
known_names = set(df['Name'])

print("Initializing BM25 Engine...")
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words]

df['tokens'] = df['Description'].apply(preprocess_text)
bm25 = BM25Okapi(df['tokens'].tolist())

print("Initializing FAISS Engine (this may take a minute)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['Description'].tolist(), show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexHNSWFlat(dimension, 32)
index.add(embeddings)

# ==========================================
# 2. SEARCH FUNCTIONS
# ==========================================
def search_games_bm25(query, top_k=5):
    query_tokens = preprocess_text(query)
    doc_scores = bm25.get_scores(query_tokens)
    df_temp = df.copy()
    df_temp['bm25_score'] = doc_scores
    return df_temp.sort_values(by='bm25_score', ascending=False).head(top_k)

def search_and_rerank_ltr(query, top_candidates=50, final_top_k=5):
    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, top_candidates)
    
    candidates = df.iloc[indices[0]].copy()
    candidates['faiss_distance'] = distances[0]
    
    scaler = MinMaxScaler()
    candidates['semantic_score'] = 1 - scaler.fit_transform(candidates[['faiss_distance']])
    candidates['norm_rating'] = scaler.fit_transform(candidates[['Rating']])
    candidates['norm_popularity'] = scaler.fit_transform(candidates[['Avg_popularity_log']])
    
    # Manual heuristic weights used for pooling
    W_SEMANTIC, W_RATING, W_POPULARITY = 0.50, 0.30, 0.20
    candidates['final_ltr_score'] = (
        (candidates['semantic_score'] * W_SEMANTIC) +
        (candidates['norm_rating'] * W_RATING) +
        (candidates['norm_popularity'] * W_POPULARITY)
    )
    
    return candidates.sort_values(by='final_ltr_score', ascending=False).head(final_top_k)

# ==========================================
# 3. HELPER FUNCTIONS FOR LLM
# ==========================================
def ranking_to_scores(ranked_names, n3=N_SCORE_3, n2=N_SCORE_2, n1=N_SCORE_1):
    scores = {}
    for i, name in enumerate(ranked_names):
        if i < n3:
            scores[name] = 3
        elif i < n3 + n2:
            scores[name] = 2
        elif i < n3 + n2 + n1:
            scores[name] = 1
        else:
            scores[name] = 0
    return scores

def parse_ranking_response(raw_text, pooled_names):
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        ranked_names = json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r'-\s*Name:\s*([^\n,]+)\n', r'"\1",\n', text)
        ranked_names = json.loads(repaired) 

    corrected = []
    for name in ranked_names:
        if name in pooled_names:
            corrected.append(name)
            continue
        match = get_close_matches(name, pooled_names, n=1, cutoff=0.75)
        corrected.append(match[0] if match else name)
    return corrected

# ==========================================
# 4. MAIN BATCH JOB EXECUTION
# ==========================================
def main():
    genai.configure(api_key=API_KEY)

    EXCLUDE_KEYWORDS = ['tts', 'embedding', 'deep-research', 'live', 'image', 'vision-only', 'audio', 'robotics']
    PREFERRED_MODELS = ['gemini-3.5-flash-lite', 'gemini-3.5-flash', 'gemini-3-flash']

    print("\nSearching for a stable, free-tier-friendly Gemini text model...")
    available_models = [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

    model_name = None
    for preferred in PREFERRED_MODELS:
        if preferred in available_models:
            model_name = preferred
            break

    if model_name is None:
        candidates = [m for m in available_models if ('flash' in m or 'pro' in m) and not any(kw in m for kw in EXCLUDE_KEYWORDS) and 'preview' not in m and 'exp' not in m and not m.startswith('gemini-2.5') and not m.startswith('gemini-2.0')]
        if candidates:
            model_name = sorted(candidates)[0]
        else:
            raise RuntimeError("No suitable stable text model found.")

    print(f"Selected model: {model_name}")
    llm = genai.GenerativeModel(model_name)

    test_queries = [
        "Cyberpunk game with hacking mechanics", "Lovecraftian horror game set in the 1920s",
        "Space exploration and empire building", "Farming simulator with animal breeding",
        "Medieval trading in the Mediterranean", "Surviving a zombie apocalypse with limited resources",
        "Pirate adventure game with ship combat", "Deck builder with worker placement",
        "Hidden movement game where one player is hunted", "Engine builder with drafting",
        "Social deduction game with hidden roles", "Tile placement game with pattern building",
        "Roll and write game with dice mitigation", "Fast paced card game for two players only",
        "Heavy economic eurogame taking over two hours", "Party game for 8 people under 30 minutes",
        "Solo game with campaign mode", "Family friendly game for young kids",
        "Quick filler game to play in 15 minutes", "A cooperative space game involving betrayal",
        "Legacy game with permanent board changes", "Miniatures skirmish game with asymmetric factions",
        "Abstract strategy game with no luck", "Dungeon crawler with character progression",
        "Cooperative detective game solving crimes"
    ]

    automated_qrels = {}
    raw_responses = {}

    if os.path.exists(qrels_file):
        with open(qrels_file, 'r') as f:
            saved = json.load(f)
        existing_qrels = saved.get("qrels", saved)
        existing_meta = saved.get("_metadata", {})
        if existing_meta.get("method") == METHOD_NAME and existing_meta.get("model") == model_name:
            automated_qrels = existing_qrels
            print(f"Found compatible checkpoint ({len(automated_qrels)} queries done). Resuming...\n")
        else:
            print("Checkpoint used different method/model. Discarding.\n")

    if os.path.exists(raw_responses_file):
        with open(raw_responses_file, 'r') as f:
            raw_responses = json.load(f)

    print("Starting Qrel generation (ranking-based)...")
    successful_calls_this_run = 0
    last_call_time = 0

    for i, query in enumerate(test_queries):
        if query in automated_qrels and len(automated_qrels[query]) > 0:
            print(f"[{i+1}/{len(test_queries)}] SKIPPED (already done): '{query}'")
            continue

        print(f"[{i+1}/{len(test_queries)}] PROCESSING: '{query}'")

        bm25_top = search_games_bm25(query, top_k=10)
        ltr_top = search_and_rerank_ltr(query, top_candidates=50, final_top_k=10)
        pool_df = pd.concat([bm25_top, ltr_top]).drop_duplicates(subset=['Name'])
        pooled_names = set(pool_df['Name'])

        games_text = ""
        for _, row in pool_df.iterrows():
            desc_snippet = str(row['Description'])[:400].replace('\n', ' ')
            games_text += f"- Name: {row['Name']} | Description: {desc_snippet}...\n"

        prompt = f"""
        You are an expert Information Retrieval evaluator and a board game domain expert.
        USER QUERY: "{query}"
        GAMES TO EVALUATE:
        {games_text}
        Rank ALL the games above from most relevant to least relevant to the query.
        Ties are allowed only when two games are truly indistinguishable in relevance.
        Output ONLY a raw JSON array of game names, ordered from most to least relevant,
        with no markdown blocks, backticks, or other text. Include every game exactly once,
        using the exact names given above.
        Example format: ["Game A", "Game B", "Game C"]
        """

        if DRY_RUN:
            continue

        elapsed = time.time() - last_call_time
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

        attempt = 0
        success = False
        while attempt < MAX_RETRIES and not success:
            attempt += 1
            try:
                response = llm.generate_content(prompt)
                last_call_time = time.time()
                successful_calls_this_run += 1

                raw_responses[query] = response.text
                with open(raw_responses_file, 'w') as f:
                    json.dump(raw_responses, f, indent=2)

                ranked_names = parse_ranking_response(response.text, pooled_names)
                missing_names = [name for name in pooled_names if name not in ranked_names]

                scores_dict = ranking_to_scores(ranked_names)
                for name in missing_names:
                    scores_dict.setdefault(name, 0)

                automated_qrels[query] = scores_dict
                run_metadata = {
                    "model": model_name,
                    "method": METHOD_NAME,
                    "last_updated": datetime.datetime.now().isoformat(),
                }
                with open(qrels_file, 'w') as f:
                    json.dump({"_metadata": run_metadata, "qrels": automated_qrels}, f, indent=4)

                success = True

            except Exception as e:
                if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                    wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    print(f"  [!] Rate limit hit. Backing off {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [!] Error on query '{query}': {e}")
                    break

if __name__ == "__main__":
    main()


# How to run?
# python generate_qrels.py