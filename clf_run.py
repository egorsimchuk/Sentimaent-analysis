from __future__ import print_function

__author__ = 'xead'
from sentiment_classifier import SentimentClassifier
from codecs import open
import time
from flask import Flask, render_template, request
app = Flask(__name__)

print("Preparing classifier")
start_time = time.time()
classifier = SentimentClassifier()

print("Classifier is ready")
print(time.time() - start_time, "seconds")

@app.route("/", methods=["POST", "GET"])
def index_page(text="", prediction_message=""):
    if request.method == "POST":
        text = request.form["text"]
    logfile = open("./logs/log.txt", "a", "utf-8")
    print(text)
    print('<response>', end="", file=logfile)
    print(text, end="", file=logfile)
    prediction_message = classifier.get_prediction_message(text)
    print(prediction_message)
    print(prediction_message, end="", file=logfile)
    print('</response>', end="", file=logfile)
    logfile.close()
	
    return render_template('test.html', text=text, prediction_message=prediction_message)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=80, debug=False)