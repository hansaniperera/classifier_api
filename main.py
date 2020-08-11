import flask
from flask import Flask, render_template, Response, request, jsonify
from graph import ImageGen

app = Flask(__name__)

@app.route('/')
def welcome():
    return "Drowning Prediction"


def gen(graph):
    while True:
        res = graph.get_prediction()
        yield res

@app.route('/prediction')
def drowning_feed():
    return Response(gen(ImageGen()))


if __name__ == '__main__':
    app.run()
