# from sklearn.ensemble import RandomForestClassifier
import pickle

# ML imports
from sklearn.naive_bayes import MultinomialNB

# from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer

from util import plot_roc

# spacy_tok


class NLPModel(object):
    """Natural Language Processing Model for sentiment analysis.

    A class that implements text vectorization and classification for sentiment analysis
    using scikit-learn's MultinomialNB classifier and TfidfVectorizer.
    """

    def __init__(self):
        """Simple NLP
        Attributes:
            clf: sklearn classifier model
            vectorizor: TFIDF vectorizer or similar
        """
        self.clf = MultinomialNB()
        # self.vectorizer = TfidfVectorizer(tokenizer=spacy_tok)
        self.vectorizer = TfidfVectorizer()

    def vectorizer_fit(self, x):
        """Fits a TFIDF vectorizer to the text"""
        self.vectorizer.fit(x)

    def vectorizer_transform(self, x):
        """Transform the text data to a sparse TFIDF matrix"""
        x_transformed = self.vectorizer.transform(x)
        return x_transformed

    def train(self, x, y):
        """Trains the classifier to associate the label with the sparse matrix"""
        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        self.clf.fit(x, y)

    def predict_proba(self, x):
        """Returns probability for the binary class '1' in a numpy array"""
        y_proba = self.clf.predict_proba(x)
        return y_proba[:, 1]

    def predict(self, x):
        """Returns the predicted class in an array"""
        y_pred = self.clf.predict(x)
        return y_pred

    def pickle_vectorizer(self, path="chalicelib/models/TFIDFVectorizer.pkl"):
        """Saves the trained vectorizer for future use."""
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
            print(f"Pickled vectorizer at {path}")

    def pickle_clf(self, path="chalicelib/models/SentimentClassifier.pkl"):
        """Saves the trained classifier for future use."""
        with open(path, "wb") as f:
            pickle.dump(self.clf, f)
            print(f"Pickled classifier at {path}")

    def plot_roc(self, x, y, size_x, size_y):
        """Plot the ROC curve for X_test and y_test."""
        plot_roc(self.clf, x, y, size_x, size_y)
