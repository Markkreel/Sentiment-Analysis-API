# Sentiment Analysis API

A machine learning API for analyzing sentiment in text phrases using Natural Language Processing (NLP) techniques.

## Features

- Binary sentiment classification (positive/negative)
- REST API endpoint for predictions
- Model training pipeline with TF-IDF vectorization
- Multinomial Naive Bayes classifier
- ROC curve evaluation metrics

## Installation

1. Clone repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- Flask
- scikit-learn
- pandas
- matplotlib

## Project Structure

```
├── app.py              # Flask API endpoints
├── build_model.py      # Model training script
├── model.py            # NLP model class definition
├── requirements.txt    # Dependency list
├── sentiment_data/     # Training data (TSV files)
└── chalicelib/models/  # Saved model artifacts
```

## Usage

1. Train the model:
   ```bash
   python build_model.py
   ```
2. Start the API:
   ```bash
   flask run
   ```

## API Endpoints

GET `/sentiment`

- Parameters:
  - `sentence`: Text phrase to analyze
- Example:
  ```
  http://localhost:5000/sentiment?sentence="I really enjoy this product"
  ```
- Response:
  ```json
  {
    "sentence": "I really enjoy this product",
    "sentiment": "positive",
    "probability": 0.92
  }
  ```

## Implementation Details

The system uses:

- TF-IDF vectorization for text feature extraction
- Multinomial Naive Bayes classifier (scikit-learn)
- ROC AUC score of 0.93 on test data
- Model artifacts persisted as pickle files
