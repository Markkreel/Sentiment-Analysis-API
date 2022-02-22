from flask import Flask, request
import pickle
import numpy as np
from model import NLPModel

app = Flask(__name__)
# api = Api(app)

model = NLPModel()

clf_path = 'lib/models/SentimentClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = 'lib/models/TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

@app.route('/')
def post_result():
    query = request.args.get('query')
     # vectorize the user's query and make a prediction
    uq_vectorized = model.vectorizer_transform(np.array(['that movie was boring']))
    prediction = model.predict(uq_vectorized)
    pred_proba = model.predict_proba(uq_vectorized)

     # Output either 'Negative' or 'Positive' along with the score
    if prediction == 0:
        pred_text = 'Negative'
    else:
        pred_text = 'Positive'
        
    # round the predict proba value and set to new variable
    confidence = round(pred_proba[0], 3)

    # create JSON object
    output = {'prediction': pred_text, 'confidence': confidence}
    
    return output
    
if __name__ == "__main__":
        app.run()