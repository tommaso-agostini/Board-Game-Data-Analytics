"""
generate_qrels_expanded.py
Batch job to generate LLM-as-a-Judge relevance scores.
Methodology: 100 queries, BM25+FAISS pure pooling, 800 chars.
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
import google.generativeai as genai

load_dotenv()

# 0. SETUP
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

API_KEY = os.environ.get("GEMINI_API_KEY", "INSERT_YOUR_API_KEY_HERE")

DRY_RUN = False
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 20
MIN_SECONDS_BETWEEN_CALLS = 4.5
METHOD_NAME = "ranking_bucket_expanded_v3"

N_SCORE_3 = 2   
N_SCORE_2 = 6   
N_SCORE_1 = 8   

QRELS_FILE = '../data/evaluation/expanded_qrels.json'
RAW_RESPONSES_FILE = '../data/evaluation/expanded_qrels_raw.json'

# 1. DATA AND ENGINE INITIALIZATION
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

print("Initializing BM25 Engine...")
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words]

df['tokens'] = df['Description'].apply(preprocess_text)
bm25 = BM25Okapi(df['tokens'].tolist())

print("Initializing FAISS Engine...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['Description'].tolist(), show_progress_bar=True).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexHNSWFlat(dimension, 32)
index.add(embeddings)

# 2. RAW RETRIEVAL FUNCTIONS 
def get_bm25_candidates(query, top_k=20):
    query_tokens = preprocess_text(query)
    doc_scores = bm25.get_scores(query_tokens)
    df_temp = df.copy()
    df_temp['bm25_score'] = doc_scores
    return df_temp.sort_values(by='bm25_score', ascending=False).head(top_k)

def get_faiss_candidates(query, top_k=20):
    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    candidates = df.iloc[indices[0]].copy()
    candidates['faiss_distance'] = distances[0]
    return candidates

# 3. STRICT LLM PARSING UTILS
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
    seen = set()
    for name in ranked_names:
        final_name = None
        if name in pooled_names:
            final_name = name
        else:
            match = get_close_matches(name, pooled_names, n=1, cutoff=0.75)
            if match:
                final_name = match[0]
                
        # Append only if it's a valid match and not already processed (deduplication)
        if final_name and final_name not in seen:
            seen.add(final_name)
            corrected.append(final_name)
            
    return corrected

# 4. MAIN BATCH JOB
def main():
    genai.configure(api_key=API_KEY)

    EXCLUDE_KEYWORDS = ['tts', 'embedding', 'deep-research', 'live', 'image', 'vision-only', 'audio', 'robotics']
    PREFERRED_MODELS = ['gemini-3.5-flash-lite','gemini-1.5-flash', 'gemini-3.5-flash', 'gemini-3-flash']

    print("\nSearching for a stable Gemini text model...")
    available_models = [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

    model_name = None
    for preferred in PREFERRED_MODELS:
        if preferred in available_models:
            model_name = preferred
            break

    if model_name is None:
        # Fallback: reverse sort to grab newer versions first if preferred list fails
        candidates = [m for m in available_models if ('flash' in m or 'pro' in m) and not any(kw in m for kw in EXCLUDE_KEYWORDS) and 'preview' not in m and 'exp' not in m]
        if candidates:
            model_name = sorted(candidates, reverse=True)[0]
        else:
            raise RuntimeError("No suitable stable text model found.")

    print(f"Selected model: {model_name}")
    llm = genai.GenerativeModel(model_name)

    test_queries = [
        # Themes & Settings
        "Cyberpunk game with hacking mechanics", "Lovecraftian horror game set in the 1920s",
        "Space exploration and empire building", "Farming simulator with animal breeding",
        "Medieval trading in the Mediterranean", "Surviving a zombie apocalypse with limited resources",
        "Pirate adventure game with ship combat", "Cold war espionage and hidden identities",
        "Building a theme park or amusement park", "Fantasy tavern management",
        "Trains and route building across America", "Dinosaur park creation and management",
        "Feudal Japan samurai conflict", "Mars colonization and terraforming",
        "Underwater city building", "Greek mythology and fighting monsters",
        "Vampire clans fighting for dominance", "Wild west shootout and outlaws",
        "Running a modern restaurant business", "Time travel and altering history",
        
        # Mechanics & Systems
        "Deck builder with worker placement", "Hidden movement game where one player is hunted",
        "Engine builder with drafting", "Social deduction game with hidden roles",
        "Tile placement game with pattern building", "Roll and write game with dice mitigation",
        "Abstract strategy game with no luck", "Dungeon crawler with character progression",
        "Cooperative detective game solving crimes", "Auction and bidding for art or goods",
        "Push your luck dice rolling", "Area control with asymmetric factions",
        "Trick taking card game", "Bag building and resource management",
        "Rondel mechanism for action selection", "Programming movement with cards",
        "Campaign game with a legacy system", "Real time cooperative chaos",
        "Story driven adventure with a spiral book", "Flicking or dexterity mechanics",
        
        # Player Counts & Targets
        "Fast paced card game for two players only", "Party game for 8 people under 30 minutes",
        "Solo game with campaign mode", "Family friendly game for young kids",
        "Quick filler game to play in 15 minutes", "Heavy economic eurogame taking over two hours",
        "Cooperative game for exactly 4 players", "Two player head to head wargame",
        "Large group trivia game", "Perfect for couples and date night",
        "Gateway game to introduce non gamers", "Highly competitive tournament game",
        "Educational game about history", "Children's game focusing on memory",
        "Scales perfectly from 1 to 5 players", "Team vs team tactical combat",
        
        # Combinations & Specific Needs
        "A cooperative space game involving betrayal", "Legacy game with permanent board changes",
        "Miniatures skirmish game with asymmetric factions", "Card drafting with set collection",
        "Economic simulation with stocks and shares", "Word game focusing on vocabulary",
        "Drawing or sketching party game", "Nature themed game about birds or trees",
        "Science fiction 4x game", "Zombie survival with a traitor mechanic",
        "Farming game without any combat", "Dice combat in a fantasy setting",
        "Escape room in a box", "Racing game with betting",
        "Civilization building with tech trees", "Trading card game with booster packs",
        "Murder mystery dinner party game", "City building with polyomino tiles",
        "Sports simulation board game", "Negotiation and trading backstabbing game",

        "Post apocalyptic survival with resource trading", "Deep sea exploration and submarine combat",
        "Cyberpunk hacking and corporate espionage", "Steampunk airship trade and combat",
        "Medieval castle siege and defense", "Vikings raiding and pillaging villages",
        "Ancient Roman gladiators in the colosseum", "Ancient Egyptian pharaohs building pyramids",
        "Wild West train robbery and heist", "Alchemical potion crafting and brewing",
        "Magical academy and wizard training", "Monster hunting and contract fulfillment",
        "Pirate treasure hunt on a deserted island", "Space station management and oxygen control",
        "Cybernetic body modification and cyberpunk street race", "Supernatural detective solving occult crimes",
        "Post apocalyptic wasteland buggy racing", "Deep space mining and mineral extraction",
        "Steampunk clockwork automaton assembly", "Medieval monastery and wine production",
        "Cooperative tower defense against hordes", "Asymmetric hidden movement in a spaceship",
        "Quick card drafting game under 20 minutes", "Heavy thematic sandbox game for solo players"
    ]

    automated_qrels = {}
    raw_responses = {}

    if os.path.exists(QRELS_FILE):
        with open(QRELS_FILE, 'r') as f:
            saved = json.load(f)
        existing_qrels = saved.get("qrels", saved)
        existing_meta = saved.get("_metadata", {})
        if existing_meta.get("method") == METHOD_NAME and existing_meta.get("model") == model_name:
            automated_qrels = existing_qrels
            print(f"Found compatible checkpoint ({len(automated_qrels)} queries done). Resuming...\n")
        else:
            print("Checkpoint used different method/model. Discarding and starting fresh.\n")

    if os.path.exists(RAW_RESPONSES_FILE):
        with open(RAW_RESPONSES_FILE, 'r') as f:
            raw_responses = json.load(f)

    print("Starting Qrel generation...")
    last_call_time = 0

    for i, query in enumerate(test_queries):
        if query in automated_qrels and len(automated_qrels[query]) > 0:
            print(f"[{i+1}/{len(test_queries)}] SKIPPED: '{query}'")
            continue

        print(f"[{i+1}/{len(test_queries)}] PROCESSING: '{query}'")

        bm25_top = get_bm25_candidates(query, top_k=20)
        faiss_top = get_faiss_candidates(query, top_k=20)
        pool_df = pd.concat([bm25_top, faiss_top]).drop_duplicates(subset=['Name'])
        pooled_names = set(pool_df['Name'])

        games_text = ""
        for _, row in pool_df.iterrows():
            desc_snippet = str(row['Description'])[:800].replace('\n', ' ')
            games_text += f"- Name: {row['Name']} | Description: {desc_snippet}...\n"

        prompt = f"""
        You are an expert Information Retrieval evaluator and a board game domain expert.
        USER QUERY: "{query}"
        GAMES TO EVALUATE:
        {games_text}
        Rank ALL the games above from most relevant to least relevant to the query.
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

                raw_responses[query] = response.text
                with open(RAW_RESPONSES_FILE, 'w') as f:
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
                with open(QRELS_FILE, 'w') as f:
                    json.dump({"_metadata": run_metadata, "qrels": automated_qrels}, f, indent=4)

                success = True

            except Exception as e:
                if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                    wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    print(f"  [!] Rate limit hit. Backing off {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [!] Fatal error on query '{query}': {e}")
                    # Log to a separate file for debugging before breaking
                    with open("fatal_errors.log", "a") as err_file:
                        err_file.write(f"Query: {query} | Error: {e}\n")
                    break

if __name__ == "__main__":
    main()