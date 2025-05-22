from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import re
import nltk
import os
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from keras.models import load_model
import traceback

# Ensure nltk data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Flask app initialization
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to the webapp directory
WEBAPP_PATH = '/Users/shivangisingh/Desktop/LexiBoost/webapp'

# ----------- Helper Functions -----------

def sent2word(x):
    stop_words = set(stopwords.words('english'))
    x = re.sub("[^A-Za-z]", " ", x)
    x = x.lower()
    words = x.split()
    filtered_sentence = [w for w in words if w not in stop_words]
    return filtered_sentence


def essay2word(essay):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw = tokenizer.tokenize(essay.strip())
    final_words = [sent2word(sentence) for sentence in raw if len(sentence) > 0]
    return final_words


def makeVec(words, model, num_features):
    vec = np.zeros((num_features,), dtype="float32")
    noOfWords = 0.
    index2word_set = set(model.index_to_key)  # Updated for newer gensim versions
    for word in words:
        if word in index2word_set:
            noOfWords += 1
            vec = np.add(vec, model[word])
    if noOfWords > 0:
        vec = np.divide(vec, noOfWords)
    return vec


def getVecs(essays, model, num_features):
    essay_vecs = np.zeros((len(essays), num_features), dtype="float32")
    for i, essay in enumerate(essays):
        essay_vecs[i] = makeVec(essay, model, num_features)
    return essay_vecs


def convertToVec(text):
    if len(text) > 20:
        try:
            num_features = 300
            word2vec_model = KeyedVectors.load_word2vec_format("word2vecmodel.bin", binary=True)

            clean_test_essays = [sent2word(text)]
            testDataVecs = getVecs(clean_test_essays, word2vec_model, num_features)

            testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

            lstm_model = load_model("final_lstm.h5")
            prediction = lstm_model.predict(testDataVecs)

            return str(round(prediction[0][0]))
        except Exception as e:
            print("Error in convertToVec:", str(e))
            traceback.print_exc()
            return "Error: " + str(e)
    else:
        return "Essay too short"


# ----------- Flask Routes -----------

# Serve the main HTML page
@app.route('/')
def serve_frontend():
    return send_from_directory(WEBAPP_PATH, 'index.html')

# Serve static files like images, css, etc.
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(WEBAPP_PATH, path)

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server is running'}), 200


@app.route('/score', methods=['POST'])
def score_essay():
    """Main endpoint for scoring essays"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        final_text = data.get("text", "")

        if not final_text or len(final_text) < 20:
            return jsonify({'error': 'Essay too short', 'score': '0'}), 400

        score = convertToVec(final_text)
        return jsonify({'score': score}), 200
    except Exception as e:
        print("Error in score_essay:", str(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# API info route
@app.route('/api', methods=['GET'])
def api_info():
    return jsonify({
        'api': 'Automated Essay Scoring API',
        'endpoints': {
            '/score': 'POST - Score an essay',
            '/health': 'GET - Check API health'
        }
    })


# ----------- Run App -----------


if __name__ == '__main__':
    print("Starting Essay Scoring API server...")
    print(f"Make sure your HTML file is at: {WEBAPP_PATH}/index.html")
    print(f"Web interface available at: http://127.0.0.1:5000")
    print(f"API info available at: http://127.0.0.1:5000/api")
    app.run(debug=True)
