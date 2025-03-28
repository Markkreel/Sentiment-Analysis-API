"""
Sentiment Analysis API

This module provides a Flask-based REST API for sentiment analysis of text.
It uses a trained machine learning model to classify text as positive or negative,
providing sentiment predictions with confidence scores.
"""

import pickle
import numpy as np
from flask import Flask, request
from model import NLPModel

app = Flask(__name__)
# api = Api(app)


class Config:
    """
    Configuration class storing file paths for the sentiment analysis model.

    Attributes:
        CLF_PATH (str): Path to the serialized sentiment classifier model
        VEC_PATH (str): Path to the serialized TF-IDF vectorizer
    """

    CLF_PATH = "lib/models/SentimentClassifier.pkl"
    VEC_PATH = "lib/models/TFIDFVectorizer.pkl"


model = NLPModel()

try:
    with open(Config.CLF_PATH, "rb") as f:
        model.clf = pickle.load(f)
    with open(Config.VEC_PATH, "rb") as f:
        model.vectorizer = pickle.load(f)
except FileNotFoundError as e:
    raise RuntimeError(f"Model file missing: {e}") from e
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}") from e


def analyze_sentiment(text):
    """
    Analyzes the sentiment of the provided text using the trained model.

    Args:
        text (str): The text to analyze for sentiment.

    Returns:
        dict: A dictionary containing:
            - prediction (str): Either "Positive" or "Negative"
            - confidence (str): Confidence score formatted to 3 decimal places

    Raises:
        ValueError: If text is empty or less than 3 characters
        RuntimeError: If there are issues with model prediction
    """
    if not text or len(text.strip()) < 3:
        raise ValueError("Query must be at least 3 characters")

    uq_vectorized = model.vectorizer_transform(np.array([text]))
    prediction = model.predict(uq_vectorized)
    pred_proba = model.predict_proba(uq_vectorized)

    return {
        "prediction": "Positive" if prediction[0] else "Negative",
        "confidence": f"{pred_proba[0]:.3f}",
    }


@app.route("/analyze")
def analyze_sentiment_route():
    """
    Flask route handler for sentiment analysis endpoint.

    Returns:
        dict: JSON response containing sentiment analysis results or error message
        int: HTTP status code

    Query Parameters:
        query (str): Text to analyze for sentiment
    """
    query = request.args.get("query")

    if not query:
        return {"error": "Missing query parameter"}, 400

    try:
        result = analyze_sentiment(query)

        return {
            "query": query,
            "result": result,
            "model": "SentimentClassifier",
            "version": "1.0",
        }
    except ValueError as e:
        return {"error": str(e)}, 400
    except (RuntimeError, pickle.PickleError) as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run()
