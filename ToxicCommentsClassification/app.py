from flask import Flask, request, Response, jsonify
import json
from datetime import datetime
from Model import GeneralModel

app = Flask(__name__)

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

@app.route('/predict/')
def get_prediction():
    review = request.args.get('review')
    if model.predict([str(review)])[0] == 0:
        return "Clean"
    else:
        return "Objectionable"


if __name__ == "__main__":
    model = GeneralModel()
    model.load_vectorizers([r'.\Pickles\tfidf_vectorizer_char.p', r'.\Pickles\tfidf_vectorizer_word.p'])
    model.load_model(r'.\Pickles\mnb.p')
    app.run(port='5000')