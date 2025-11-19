import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import json
from pathlib import Path

# Import the app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from main import app

client = TestClient(app)


@pytest.fixture
def sample_tiktok_data():
    """Sample TikTok JSON data for testing"""
    return {
        "Your Activity": {
            "Hashtag": {
                "HashtagList": [
                    {"HashtagName": "technology"},
                    {"HashtagName": "coding"},
                    {"HashtagName": "ai"}
                ]
            }
        }
    }


@pytest.fixture
def sample_newsapi_response():
    """Mock NewsAPI response"""
    return {
        "status": "ok",
        "totalResults": 3,
        "articles": [
            {
                "title": "AI Breakthrough in Machine Learning",
                "description": "Scientists develop new AI model",
                "url": "https://example.com/ai-news",
                "urlToImage": None,
                "publishedAt": "2024-01-01T00:00:00Z"
            },
            {
                "title": "Latest Tech Trends",
                "description": "Technology industry updates",
                "url": "https://example.com/tech-news",
                "urlToImage": None,
                "publishedAt": "2024-01-01T00:00:00Z"
            },
            {
                "title": "Programming Languages Update",
                "description": "New features in popular languages",
                "url": "https://example.com/programming-news",
                "urlToImage": None,
                "publishedAt": "2024-01-01T00:00:00Z"
            }
        ]
    }


def test_root_endpoint():
    """Test that the root endpoint returns the HTML page"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "TikTok" in response.text or "News Recommender" in response.text


@patch('main.requests.get')
def test_recommend_success(mock_get, sample_tiktok_data, sample_newsapi_response):
    """Test successful recommendation request"""
    # Mock the NewsAPI response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_newsapi_response
    mock_response.text = json.dumps(sample_newsapi_response)
    mock_get.return_value = mock_response

    # Create a file-like object for the TikTok data
    files = {
        'file': ('test.json', json.dumps(sample_tiktok_data), 'application/json')
    }
    data = {
        'news_api_key': 'test-api-key'
    }

    response = client.post("/recommend", files=files, data=data)
    
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert len(result) > 0
    
    # Check that results have required fields
    for article in result:
        assert "title" in article
        assert "description" in article
        assert "url" in article
        assert "score" in article
        assert isinstance(article["score"], (int, float))
    
    # Check that results are sorted by score (descending)
    scores = [article["score"] for article in result]
    assert scores == sorted(scores, reverse=True)


def test_recommend_invalid_json():
    """Test recommendation with invalid JSON file"""
    files = {
        'file': ('test.json', 'not valid json', 'application/json')
    }
    data = {
        'news_api_key': 'test-api-key'
    }

    response = client.post("/recommend", files=files, data=data)
    assert response.status_code == 400
    assert "Invalid JSON file" in response.json()["detail"]


def test_recommend_no_hashtags():
    """Test recommendation with TikTok data containing no hashtags"""
    invalid_data = {
        "Your Activity": {
            "Hashtag": {
                "HashtagList": []
            }
        }
    }

    files = {
        'file': ('test.json', json.dumps(invalid_data), 'application/json')
    }
    data = {
        'news_api_key': 'test-api-key'
    }

    response = client.post("/recommend", files=files, data=data)
    assert response.status_code == 400
    assert "No hashtags found" in response.json()["detail"]


def test_recommend_empty_hashtag_names():
    """Test recommendation with hashtags but no HashtagName fields"""
    invalid_data = {
        "Your Activity": {
            "Hashtag": {
                "HashtagList": [
                    {"HashtagName": ""},
                    {"HashtagName": "   "}
                ]
            }
        }
    }

    files = {
        'file': ('test.json', json.dumps(invalid_data), 'application/json')
    }
    data = {
        'news_api_key': 'test-api-key'
    }

    response = client.post("/recommend", files=files, data=data)
    assert response.status_code == 400
    assert "No usable hashtag texts detected" in response.json()["detail"]


@patch('main.requests.get')
def test_recommend_newsapi_error(mock_get, sample_tiktok_data):
    """Test recommendation when NewsAPI returns an error"""
    # Mock NewsAPI error response
    mock_response = Mock()
    mock_response.status_code = 401
    mock_response.text = "Invalid API key"
    mock_get.return_value = mock_response

    files = {
        'file': ('test.json', json.dumps(sample_tiktok_data), 'application/json')
    }
    data = {
        'news_api_key': 'invalid-key'
    }

    response = client.post("/recommend", files=files, data=data)
    assert response.status_code == 500
    assert "NewsAPI error" in response.json()["detail"]


@patch('main.requests.get')
def test_recommend_no_articles(mock_get, sample_tiktok_data):
    """Test recommendation when NewsAPI returns no articles"""
    # Mock NewsAPI response with no articles
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "ok",
        "totalResults": 0,
        "articles": []
    }
    mock_response.text = json.dumps({"articles": []})
    mock_get.return_value = mock_response

    files = {
        'file': ('test.json', json.dumps(sample_tiktok_data), 'application/json')
    }
    data = {
        'news_api_key': 'test-api-key'
    }

    response = client.post("/recommend", files=files, data=data)
    assert response.status_code == 500
    assert "No news articles returned" in response.json()["detail"]


@patch('main.requests.get')
def test_recommend_missing_news_api_key(mock_get, sample_tiktok_data, sample_newsapi_response):
    """Test recommendation without providing news_api_key"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_newsapi_response
    mock_response.text = json.dumps(sample_newsapi_response)
    mock_get.return_value = mock_response

    files = {
        'file': ('test.json', json.dumps(sample_tiktok_data), 'application/json')
    }
    data = {}

    response = client.post("/recommend", files=files, data=data)
    # FastAPI should return 422 for missing required form field
    assert response.status_code == 422


@patch('main.requests.get')
def test_recommend_articles_with_missing_fields(mock_get, sample_tiktok_data):
    """Test that articles with missing title/description are filtered out"""
    # Mock NewsAPI response with some articles missing required fields
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "ok",
        "totalResults": 3,
        "articles": [
            {
                "title": "Valid Article",
                "description": "Has description",
                "url": "https://example.com/valid",
                "urlToImage": None,
                "publishedAt": "2024-01-01T00:00:00Z"
            },
            {
                "title": None,  # Missing title
                "description": "No title",
                "url": "https://example.com/invalid1",
                "urlToImage": None,
                "publishedAt": "2024-01-01T00:00:00Z"
            },
            {
                "title": "No Description",
                "description": None,  # Missing description
                "url": "https://example.com/invalid2",
                "urlToImage": None,
                "publishedAt": "2024-01-01T00:00:00Z"
            }
        ]
    }
    mock_response.text = json.dumps(mock_response.json.return_value)
    mock_get.return_value = mock_response

    files = {
        'file': ('test.json', json.dumps(sample_tiktok_data), 'application/json')
    }
    data = {
        'news_api_key': 'test-api-key'
    }

    response = client.post("/recommend", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    # Should only return the article with both title and description
    assert len(result) == 1
    assert result[0]["title"] == "Valid Article"

