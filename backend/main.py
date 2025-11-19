from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd, requests, json

app = FastAPI()

# Serve frontend files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    # Serve main HTML page
    return FileResponse("frontend/index.html")


# Load ML model
model = SentenceTransformer("all-MiniLM-L6-v2")


@app.post("/recommend")
async def recommend(file: UploadFile, news_api_key: str = Form(...)):

    # Load TikTok JSON and extract hashtags
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

    texts = [h.get("HashtagName", "").strip()
             for h in hashtag_list if h.get("HashtagName")]

    if not texts:
        raise HTTPException(status_code=400, detail="No usable hashtag texts detected")


    # Embed TikTok hashtags → interest vector
    hashtag_embs = model.encode(texts, show_progress_bar=False)
    profile_vec = hashtag_embs.mean(axis=0)


    # 3. Fetch up to 100 news articles
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


    # Embed all article texts
    news_texts = (df["title"] + " " + df["description"]).tolist()
    news_embs = model.encode(news_texts, show_progress_bar=False)

    scores = cosine_similarity(news_embs, profile_vec.reshape(1, -1)).flatten()
    df["score"] = scores


    # Return articles sorted by similarity
    df = df.sort_values("score", ascending=False)

    return df[["title", "description", "url", "score"]].to_dict(orient="records")
