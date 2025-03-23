#app.py
from flask import Flask, request, jsonify
from question import Question
import os
from dotenv import load_dotenv
from anaytical_report import generate_report

load_dotenv()

app = Flask(__name__)

reg_question = Question()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if 'question' in data:
        try:
            question = data['question']
            rag_response = reg_question.ask_question(question)  # ✅ Corrected variable name
            return jsonify({'answer': rag_response})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'No question provided'}), 400

@app.route('/analytics', methods=['POST'])
def analytics():
    try:
        report = generate_report()
        return jsonify({'report': report}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
