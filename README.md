# TikTok Interest–Based News Recommender

This project analyzes TikTok hashtag data and recommends recent news articles that match the user's interests. It uses a transformer-based text embedding model to compare TikTok-derived interest vectors with current news headlines.

## Overview

The system performs the following steps:

1. Loads a TikTok JSON file containing the user's hashtag history.
2. Extracts hashtag names and treats them as text inputs.
3. Uses a sentence-transformer model to embed the hashtags into vector space.
4. Fetches up to 100 recent news articles from NewsAPI.
5. Embeds articles using the same model.
6. Computes cosine similarity between the user's interest vector and each article.
7. Returns all articles ranked by similarity.
8. Displays results in a simple HTML/JavaScript front end.

## Technologies Used

* Python
* FastAPI (backend web framework)
* SentenceTransformers (machine learning embeddings)
* NewsAPI (news data source)
* HTML and JavaScript (frontend)
* Uvicorn (ASGI server)

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Navigate into the `backend` directory:

   ```bash
   cd backend
   ```

3. Start the server:

   ```bash
   uvicorn main:app --reload
   ```

4. Open the web interface in a browser:

   ```
   http://127.0.0.1:8000/
   ```

5. Upload:

   * Your TikTok JSON file
   * Your NewsAPI key

6. View the recommended news articles sorted by similarity.

## Project Structure

```
NewsReccomendation/
│
├── backend/
│   └── main.py        # FastAPI backend, ML logic, and news ranking
├── frontend/
│   └── index.html     # Browser interface
├── sample_data/       # Sample TikTok JSON files for testing
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## TikTok Input Format

The system expects the input file to follow this structure:

```json
{
  "Your Activity": {
    "Hashtag": {
      "HashtagList": [
        { "HashtagName": "example" }
      ]
    }
  }
}
```

## Output

The `/recommend` endpoint returns a list of news articles in JSON format. Each entry includes:

* `title`
* `description`
* `url`
* `score` (cosine similarity to user interests)

The frontend displays a subset of these results.

## Purpose

This project demonstrates:

* Text vectorization using transformer embeddings
* Unsupervised machine learning for similarity and recommendation
* API integration and data processing
* Building an end-to-end pipeline from raw data to a usable interface
