"""Utility functions for sentiment analysis and model evaluation.

This module provides helper functions for text processing using spaCy
and visualization tools for model evaluation including ROC curve plotting.
"""

# import spacy
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# nlp = spacy.load('en')


# def spacy_tok(text, lemmatize=False):
#     doc = nlp(text)
#     if lemmatize:
#         tokens = [tok.lemma_ for tok in doc]
#     else:
#         tokens = [tok.text for tok in doc]
#     return tokens


def plot_roc(model, x_columns, y_true, size_x=12, size_y=12):
    """Returns a ROC plot

    Forked from Matt Drury.
    """

    y_pred = model.predict_proba(x_columns)

    fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
    area_under_curve = auc(fpr, tpr)

    # method I: plt
    _, ax = plt.subplots(figsize=(size_x, size_y))
    model_name = str(type(model)).rsplit(".", maxsplit=1)[-1].strip(">'")
    plt.title(f"{model_name} ROC")
    ax.plot(fpr, tpr, "k", label=f"AUC = {area_under_curve:.3f}")

    ax.legend(loc="lower right")
    ax.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()
