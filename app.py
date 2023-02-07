import os
import json
import numpy as np
import yaml
import openai
import QuestionAnswerBot 
# importlib.reload(QuestionAnswerBot)

from flask import Flask, request, url_for, redirect, render_template, session
from flask_session import Session
app = Flask(__name__)
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

@app.route('/')
def hello_world():
    # load config file and set open AI API key
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    openai.api_key = config["OPENAI_API_KEY"]
    return render_template("customer_review_chatbot.html")

@app.route('/load', methods=["POST", "GET"])
def load():
    # get business/proudct name
    name = [x for x in request.form.values()][0]
    place_id_mapping = {"yelp-immersion-spa": "oBtORnu25mYpaS8eJQY_kQ",
                        "yelp-wasabi-sushi": "cFTq5MmBDPb_VEqQXPw2Dw"}
    # place_id = "oBtORnu25mYpaS8eJQY_kQ"
    place_id = place_id_mapping[name]

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # instantiate QuestionAnswerBot class
    qabot = QuestionAnswerBot.Query(place_id=place_id)

    # specify data paths and load data into the class
    embedding_path = os.path.join(config["EMBEDDINGS_PATH"], f"{place_id}_embeddings.json")
    review_path = os.path.join(config["REVIEWS_PATH"], f"{place_id}_reviews.json")
    qabot.load_data(embedding_path=embedding_path, review_path=review_path)

    # save the loaded object in flask session
    session['qabot'] = qabot
    return render_template('customer_review_chatbot.html', load=f"Loaded customer reviews from {name}.")

@app.route('/answer', methods=["POST", "GET"])
def answer():
    # get query
    query = [x for x in request.form.values()][0]

    # ask query as question
    qabot = session.get('qabot', None)
    answer = qabot.answer_query(query, show_prompt=True)

    # replace newline characters
    answer["prompt"]["context"] = answer["prompt"]["context"].replace('\n', '<br>')

    return render_template('customer_review_chatbot.html', 
                           answer=f"{answer['response']}", 
                           context=f"Context:<br>{answer['prompt']['context']}")

if __name__ == '__main__':
    app.run(debug=True)