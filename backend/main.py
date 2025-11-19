from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import requests
import json
import re
from urllib.parse import quote_plus

app = FastAPI()

# Serve the frontend files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
async def root():
    return FileResponse("../frontend/index.html")

# Load ML model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def clean_hashtag(name: str) -> str:
    """Basic cleaning: lowercase, keep letters/numbers only."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name


@app.post("/recommend")
async def recommend(file: UploadFile, news_api_key: str = Form(...)):

    # 1. Load TikTok JSON and extract hashtags
    raw_bytes = await file.read()
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    hashtag_list = (
        raw.get("Your Activity", {})
           .get("Hashtag", {})
           .get("HashtagList", [])
    )

    if not hashtag_list:
        raise HTTPException(status_code=400, detail="No hashtags found in TikTok file")

    # Original hashtag texts
    raw_tags = [h.get("HashtagName", "").strip()
                for h in hashtag_list if h.get("HashtagName")]
    if not raw_tags:
        raise HTTPException(status_code=400, detail="No usable hashtag texts detected")

    # 2. Build texts for interest vector (use original tag strings)
    texts = raw_tags

    hashtag_embs = model.encode(texts, show_progress_bar=False)
    profile_vec = hashtag_embs.mean(axis=0)

    # 3. Build a NewsAPI search query from cleaned hashtags
    #    Filter out generic/noisy ones and very short ones
    stop_tags = {
        "fyp", "foryou", "trending", "viral", "funny", "explore",
        "tiktok", "tiktokdance", "xyzbca"
    }

    cleaned_tags = []
    for t in raw_tags:
        ct = clean_hashtag(t)
        if not ct:
            continue
        if ct in stop_tags:
            continue
        # ignore very short tokens (1–2 chars)
        if len(ct) < 3:
            continue
        cleaned_tags.append(ct)

    # Deduplicate and limit how many we put in the query
    cleaned_tags = list(dict.fromkeys(cleaned_tags))[:5]

    # If nothing meaningful remains, fall back to a generic query
    if cleaned_tags:
        query = " OR ".join(cleaned_tags)
    else:
        query = ""  # will just pull general articles

    # 4. Fetch up to 100 news articles using the query
    if query:
        q_param = quote_plus(query)
        url = (
            "https://newsapi.org/v2/everything?"
            f"q={q_param}&language=en&pageSize=100&page=1&apiKey={news_api_key}"
        )
    else:
        # fallback: generic headlines if no good hashtags to query
        url = (
            "https://newsapi.org/v2/top-headlines?"
            f"language=en&pageSize=100&page=1&apiKey={news_api_key}"
        )

    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"NewsAPI error: {response.text}"
        )

    data = response.json()
    articles = data.get("articles", [])
    if not articles:
        raise HTTPException(status_code=500, detail="No news articles returned")

    df = pd.DataFrame(articles).dropna(subset=["title", "description"])
    if df.empty:
        raise HTTPException(status_code=500, detail="No usable news articles returned")

    # 5. Embed all article texts
    news_texts = (df["title"] + " " + df["description"]).tolist()
    news_embs = model.encode(news_texts, show_progress_bar=False)

    scores = cosine_similarity(news_embs, profile_vec.reshape(1, -1)).flatten()
    df["score"] = scores

    # 6. Sort by similarity and return all
    df = df.sort_values("score", ascending=False)

    return df[["title", "description", "url", "score"]].to_dict(orient="records")
