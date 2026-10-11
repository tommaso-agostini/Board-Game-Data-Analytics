"""Qrels v4 — natural-text retrieval pool and LLM-as-a-Judge labels.

Pipeline:
    NB6 -> data/tokens.pkl
    NB7 -> data/embeddings_nat.npy
    this script -> BM25 top-20 UNION dense top-20 -> LLM judge -> qrels v4

This script does NOT regenerate BM25 tokens or document embeddings.
It also does NOT read or modify the old v3 qrels files.
"""

import ast
import datetime
import html
import json
import os
import re
import string
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from google import genai
from google.genai import types


# ------------------------------------------------------------------
# CONFIG
DATA_DIR = Path("../data")
DETAILS_PATH = DATA_DIR / "details.csv"

TOKENS_PATH = DATA_DIR / "tokens.pkl"
EMB_NAT_PATH = DATA_DIR / "embeddings_nat.npy"

QRELS_FILE = DATA_DIR / "evaluation" / "expanded_qrels_v4.json"
RAW_RESPONSES_FILE = DATA_DIR / "evaluation" / "expanded_qrels_v4_raw.json"

METHOD_NAME = "ranking_bucket_expanded_v4_ids"

# Set GEMINI_MODEL in .env to override this.
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

PILOT = False
PILOT_STEP = 5

POOL_K = 20
DESC_CHARS = 800
USE_METADATA = True
MIN_COVERAGE = 0.9

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 20
MIN_SECONDS_BETWEEN_CALLS = 4.5

N_SCORE_3 = 2
N_SCORE_2 = 6
N_SCORE_1 = 8

QUERIES = ['Cyberpunk game with hacking mechanics', 'Lovecraftian horror game set in the 1920s', 'Space exploration and empire building', 'Farming simulator with animal breeding', 'Medieval trading in the Mediterranean', 'Surviving a zombie apocalypse with limited resources', 'Pirate adventure game with ship combat', 'Cold war espionage and hidden identities', 'Building a theme park or amusement park', 'Fantasy tavern management', 'Trains and route building across America', 'Dinosaur park creation and management', 'Feudal Japan samurai conflict', 'Mars colonization and terraforming', 'Underwater city building', 'Greek mythology and fighting monsters', 'Vampire clans fighting for dominance', 'Wild west shootout and outlaws', 'Running a modern restaurant business', 'Time travel and altering history', 'Deck builder with worker placement', 'Hidden movement game where one player is hunted', 'Engine builder with drafting', 'Social deduction game with hidden roles', 'Tile placement game with pattern building', 'Roll and write game with dice mitigation', 'Abstract strategy game with no luck', 'Dungeon crawler with character progression', 'Cooperative detective game solving crimes', 'Auction and bidding for art or goods', 'Push your luck dice rolling', 'Area control with asymmetric factions', 'Trick taking card game', 'Bag building and resource management', 'Rondel mechanism for action selection', 'Programming movement with cards', 'Campaign game with a legacy system', 'Real time cooperative chaos', 'Story driven adventure with a spiral book', 'Flicking or dexterity mechanics', 'Fast paced card game for two players only', 'Party game for 8 people under 30 minutes', 'Solo game with campaign mode', 'Family friendly game for young kids', 'Quick filler game to play in 15 minutes', 'Heavy economic eurogame taking over two hours', 'Cooperative game for exactly 4 players', 'Two player head to head wargame', 'Large group trivia game', 'Perfect for couples and date night', 'Gateway game to introduce non gamers', 'Highly competitive tournament game', 'Educational game about history', "Children's game focusing on memory", 'Scales perfectly from 1 to 5 players', 'Team vs team tactical combat', 'A cooperative space game involving betrayal', 'Legacy game with permanent board changes', 'Miniatures skirmish game with asymmetric factions', 'Card drafting with set collection', 'Economic simulation with stocks and shares', 'Word game focusing on vocabulary', 'Drawing or sketching party game', 'Nature themed game about birds or trees', 'Science fiction 4x game', 'Zombie survival with a traitor mechanic', 'Farming game without any combat', 'Dice combat in a fantasy setting', 'Escape room in a box', 'Racing game with betting', 'Civilization building with tech trees', 'Trading card game with booster packs', 'Murder mystery dinner party game', 'City building with polyomino tiles', 'Sports simulation board game', 'Negotiation and trading backstabbing game', 'Post apocalyptic survival with resource trading', 'Deep sea exploration and submarine combat', 'Cyberpunk hacking and corporate espionage', 'Steampunk airship trade and combat', 'Medieval castle siege and defense', 'Vikings raiding and pillaging villages', 'Ancient Roman gladiators in the colosseum', 'Ancient Egyptian pharaohs building pyramids', 'Wild West train robbery and heist', 'Alchemical potion crafting and brewing', 'Magical academy and wizard training', 'Monster hunting and contract fulfillment', 'Pirate treasure hunt on a deserted island', 'Space station management and oxygen control', 'Cybernetic body modification and cyberpunk street race', 'Supernatural detective solving occult crimes', 'Post apocalyptic wasteland buggy racing', 'Deep space mining and mineral extraction', 'Steampunk clockwork automaton assembly', 'Medieval monastery and wine production', 'Cooperative tower defense against hordes', 'Asymmetric hidden movement in a spaceship', 'Quick card drafting game under 20 minutes', 'Heavy thematic sandbox game for solo players']


# ------------------------------------------------------------------
# HELPERS
def clean_html(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


def parse_list(value):
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value]
    if isinstance(parsed, (list, tuple)):
        return [str(x) for x in parsed]
    return [str(parsed)]


def build_meta(row):
    parts = []

    mn, mx = row.get("minplayers"), row.get("maxplayers")
    if pd.notna(mn) and pd.notna(mx):
        parts.append(f"players {int(mn)}-{int(mx)}")

    pt = row.get("playingtime")
    if pd.notna(pt) and pt > 0:
        parts.append(f"{int(pt)} min")

    cats = parse_list(row.get("boardgamecategory"))[:6]
    mechs = parse_list(row.get("boardgamemechanic"))[:8]

    if cats:
        parts.append("categories: " + ", ".join(cats))
    if mechs:
        parts.append("mechanics: " + ", ".join(mechs))

    return "; ".join(parts)


def make_prompt(query, games_text):
    return f"""You are an expert Information Retrieval evaluator and a board game domain expert.

USER QUERY: "{query}"

GAMES TO EVALUATE (one per line: ID, name, game info, description excerpt):

{games_text}

Rank ALL the games above from most relevant to least relevant to the query.

Judge only how well each game matches what the query asks for (theme, mechanics,
player count, length, complexity), using the information given. Do not favor a
game just because it is famous.

Output ONLY a raw JSON array of the game IDs (integers), ordered from most to
least relevant, with no markdown blocks, backticks, names or other text.

Include every ID exactly once.

Example format: [101, 7, 342]"""


def parse_ranked_ids(raw_text, pool_ids):
    # Accept a JSON array even if the model adds a short prefix/suffix.
    match = re.search(r"\[[^\[\]]*\]", raw_text or "")
    if not match:
        return []

    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    ranked = []
    seen = set()
    for item in items:
        try:
            gid = int(item)
        except (TypeError, ValueError):
            continue
        if gid in pool_ids and gid not in seen:
            seen.add(gid)
            ranked.append(gid)

    return ranked


def ranking_to_scores(ranked_ids):
    scores = {}
    for i, gid in enumerate(ranked_ids):
        if i < N_SCORE_3:
            scores[gid] = 3
        elif i < N_SCORE_3 + N_SCORE_2:
            scores[gid] = 2
        elif i < N_SCORE_3 + N_SCORE_2 + N_SCORE_1:
            scores[gid] = 1
        else:
            scores[gid] = 0
    return scores


# ------------------------------------------------------------------
# DATA / RETRIEVAL SETUP
def setup():
    import pickle
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    for pkg in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(pkg, quiet=True)

    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):
        if not isinstance(text, str):
            return []
        text = text.lower().translate(
            str.maketrans("", "", string.punctuation)
        )
        return [w for w in word_tokenize(text) if w not in stop_words]

    print("Loading datasets...")

    df_clean = pd.read_csv(DATA_DIR / "cleaned_dataset.csv")
    df_raw = pd.read_csv(DATA_DIR / "DM1_game_dataset.csv")

    df = pd.merge(
        df_clean,
        df_raw[["BGGId", "Description"]],
        on="BGGId",
        how="inner",
    )
    df = df.dropna(subset=["Description"]).reset_index(drop=True)

    # Natural-language source from details.csv, joined by stable BGGId.
    orig = pd.read_csv(DETAILS_PATH).drop_duplicates("id")
    meta_df = df[["BGGId"]].merge(
        orig,
        left_on="BGGId",
        right_on="id",
        how="left",
    )

    if len(meta_df) != len(df):
        raise ValueError("details.csv merge changed the number of rows.")

    nat = meta_df["description"].map(clean_html)
    has_orig = nat.str.len() > 0

    missing = int((~has_orig).sum())
    print(f"Games without original text (fallback to Description): {missing} / {len(df)}")

    df["Description_nat"] = nat.where(has_orig, df["Description"])
    df["meta"] = (
        meta_df.apply(build_meta, axis=1) if USE_METADATA else ""
    )

    # NB6 artifact: do not regenerate it here.
    if not TOKENS_PATH.exists():
        raise FileNotFoundError(
            f"{TOKENS_PATH} not found. Run NB6 first."
        )

    with open(TOKENS_PATH, "rb") as f:
        tokens = pickle.load(f)

    if len(tokens) != len(df):
        raise ValueError(
            f"tokens.pkl has {len(tokens)} entries, but dataframe has {len(df)} rows. "
            "Regenerate tokens.pkl by running NB6."
        )

    bm25 = BM25Okapi(tokens)

    # NB7 artifact: do not regenerate it here.
    if not EMB_NAT_PATH.exists():
        raise FileNotFoundError(
            f"{EMB_NAT_PATH} not found. Run NB7 first."
        )

    emb = np.load(EMB_NAT_PATH).astype("float32")

    if emb.ndim != 2 or emb.shape[0] != len(df):
        raise ValueError(
            f"embeddings_nat.npy shape={emb.shape}, expected first dimension {len(df)}. "
            "Regenerate embeddings_nat.npy by running NB7."
        )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return {
        "df": df,
        "bm25": bm25,
        "model": model,
        "emb": emb,
        "preprocess": preprocess_text,
    }


def build_pool(query, ctx, k=POOL_K):
    """BM25 top-k UNION dense top-k, then deterministic shuffle."""
    bm_scores = ctx["bm25"].get_scores(ctx["preprocess"](query))

    query_vec = ctx["model"].encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")[0]

    # Exact Euclidean distance. This matches the previous v4 design.
    distances = np.linalg.norm(ctx["emb"] - query_vec, axis=1)

    bm_idx = np.argsort(-bm_scores)[:k]
    dense_idx = np.argsort(distances)[:k]

    pool_idx = np.union1d(bm_idx, dense_idx)

    rng = np.random.default_rng(zlib.crc32(query.encode("utf-8")))
    return rng.permutation(pool_idx)


def games_block(pool_idx, df):
    lines = []

    for i in pool_idx:
        row = df.iloc[i]
        desc = str(row["Description_nat"])[:DESC_CHARS]
        info = f" | Info: {row['meta']}" if row["meta"] else ""

        lines.append(
            f"- ID: {int(row['BGGId'])} | "
            f"Name: {row['Name']}{info} | "
            f"Description: {desc}..."
        )

    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------
# OUTPUT / GEMINI
def save_outputs(qrels, names, stats, raw_responses, model_name):
    QRELS_FILE.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "_metadata": {
            "model": model_name,
            "method": METHOD_NAME,
            "pool_k": POOL_K,
            "desc_chars": DESC_CHARS,
            "use_metadata": USE_METADATA,
            "min_coverage": MIN_COVERAGE,
            "pilot": PILOT,
            "last_updated": datetime.datetime.now().isoformat(),
        },
        "qrels": qrels,
        "names": names,
        "_stats": stats,
    }

    with open(QRELS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)

    with open(RAW_RESPONSES_FILE, "w", encoding="utf-8") as f:
        json.dump(raw_responses, f, indent=2, ensure_ascii=False)


def main():
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Put it in .env or export it in the shell."
        )

    queries = list(QUERIES)
    if PILOT:
        queries = queries[::PILOT_STEP]

    print(f"{len(queries)} queries to judge (PILOT={PILOT}).")

    ctx = setup()
    df = ctx["df"]
    id2name = dict(zip(df["BGGId"].astype(int), df["Name"]))

    client = genai.Client(api_key=api_key)
    model_name = MODEL_NAME
    print(f"Selected model: {model_name}")

    # Resume only from the NEW v4 output. No v3 qrels are read.
    qrels = {}
    names = {}
    stats = {}
    raw_responses = {}

    if QRELS_FILE.exists():
        with open(QRELS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)

        meta = saved.get("_metadata", {})

        if (
            meta.get("method") != METHOD_NAME
            or meta.get("model") != model_name
        ):
            raise RuntimeError(
                f"{QRELS_FILE} exists but was produced with another "
                f"method/model ({meta.get('method')} / {meta.get('model')}). "
                "Delete or rename the file before starting a new run."
            )

        qrels = saved.get("qrels", {})
        names = saved.get("names", {})
        stats = saved.get("_stats", {})
        print(f"Resuming: {len(qrels)} queries already completed.")

    if RAW_RESPONSES_FILE.exists():
        with open(RAW_RESPONSES_FILE, "r", encoding="utf-8") as f:
            raw_responses = json.load(f)

    last_call = 0.0

    for n, query in enumerate(queries, start=1):
        if query in qrels:
            print(f"[{n}/{len(queries)}] SKIPPED: '{query}'")
            continue

        print(f"[{n}/{len(queries)}] PROCESSING: '{query}'")

        pool_idx = build_pool(query, ctx)
        pool_ids = {int(df.iloc[i]["BGGId"]) for i in pool_idx}

        prompt = make_prompt(
            query,
            games_block(pool_idx, df),
        )

        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - last_call)
            if wait > 0:
                time.sleep(wait)

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                    ),
                )

                last_call = time.time()
                raw_text = response.text or ""
                raw_responses[query] = raw_text

                # Save raw response immediately, including failed/incomplete calls.
                RAW_RESPONSES_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(RAW_RESPONSES_FILE, "w", encoding="utf-8") as f:
                    json.dump(raw_responses, f, indent=2, ensure_ascii=False)

                ranked = parse_ranked_ids(raw_text, pool_ids)
                coverage = len(ranked) / len(pool_ids) if pool_ids else 0.0

                if coverage < MIN_COVERAGE:
                    print(
                        f"  [!] only {len(ranked)}/{len(pool_ids)} valid IDs "
                        f"({coverage:.1%}), attempt {attempt}/{MAX_RETRIES}"
                    )
                    continue

                scores = ranking_to_scores(ranked)

                # Only ranked games receive scores. Missing IDs remain unjudged.
                qrels[query] = {
                    str(gid): score
                    for gid, score in scores.items()
                }

                names.update({
                    str(gid): id2name[gid]
                    for gid in scores
                    if gid in id2name
                })

                stats[query] = {
                    "n_pool": len(pool_ids),
                    "n_ranked": len(ranked),
                    "coverage": coverage,
                }

                save_outputs(
                    qrels,
                    names,
                    stats,
                    raw_responses,
                    model_name,
                )

                success = True
                break

            except Exception as exc:
                msg = str(exc)
                if (
                    "429" in msg
                    or "RESOURCE_EXHAUSTED" in msg.upper()
                    or "rate limit" in msg.lower()
                ):
                    backoff = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    print(
                        f"  [!] Rate limit. Backing off {backoff}s "
                        f"(attempt {attempt}/{MAX_RETRIES})..."
                    )
                    time.sleep(backoff)
                else:
                    print(f"  [!] Error on '{query}': {exc}")
                    with open("fatal_errors.log", "a", encoding="utf-8") as err:
                        err.write(f"Query: {query} | Error: {exc}\n")
                    break

        if not success:
            print(f"  [X] query skipped: '{query}'")

    print(
        f"Done. Completed {len(qrels)}/{len(queries)} queries. "
        f"Qrels: {QRELS_FILE}"
    )


if __name__ == "__main__":
    main()
