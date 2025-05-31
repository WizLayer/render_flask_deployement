from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel
import joblib
import pickle
import numpy as np
# from nltk.stem import WordNetLemmatizer
# import nltk

# nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

# class LemmaTokenizer:
#     def __init__(self):
#         self.wnl = WordNetLemmatizer()
#     def __call__(self, doc):
#         return [self.wnl.lemmatize(t) for t in doc.split()]
    
from tokenizer import LemmaTokenizer

model = joblib.load('activity_pipeline.pkl')

model.predict(['go for a jog and go to schoool'])

@app.route('/predict', methods = ['POST'])
def predict():
  data = request.json
  text = data.get('text', '')

  if not text:
    return jsonify({'error': 'No text provided'}), 400

  prediction = model.predict([text])

  print(prediction[0])
  return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
  app.run()