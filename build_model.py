"""
This module handles the building and training of a sentiment analysis model.
It includes functionality for data loading, preprocessing, model training,
and evaluation using ROC curves.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from model import NLPModel


def build_model():
    """
    Builds and trains a sentiment analysis model using NLP techniques.

    This function:
    1. Loads training data from a TSV file
    2. Preprocesses the data for binary sentiment classification
    3. Fits and transforms the text data using vectorization
    4. Trains a classifier model
    5. Saves the trained model and vectorizer
    6. Plots the ROC curve for model evaluation

    Returns:
        None
    """
    model = NLPModel()

    # filename = os.path.join(
    #     os.path.dirname(__file__), 'chalicelib', 'all/train.tsv')
    with open("../sentiment_data/train.tsv", encoding="utf-8") as f:
        data = pd.read_csv(f, sep="\t")

    pos_neg = data[(data["Sentiment"] == 0) | (data["Sentiment"] == 4)]

    pos_neg["Binary"] = pos_neg.apply(lambda x: 0 if x["Sentiment"] == 0 else 1, axis=1)

    model.vectorizer_fit(pos_neg.loc[:, "Phrase"])
    print("Vectorizer fit complete")

    x = model.vectorizer_transform(pos_neg.loc[:, "Phrase"])
    print("Vectorizer transform complete")
    y = pos_neg.loc[:, "Binary"]

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    model.train(x_train, y_train)
    print("Model training complete")

    model.pickle_clf()
    model.pickle_vectorizer()

    model.plot_roc(x_test, y_test, size_x=10, size_y=6)


if __name__ == "__main__":
    build_model()
