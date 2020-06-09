from flask import Flask, render_template, request, json

from knn import KNN_NLC_Classifer
from cnn_tweet_classifier import cnn_classifier
from naive_bayes_tweet_classifier import naive_bayes_tweet_classifier

categoryLabels = ["TV&Movies", "Music", "Health", "Religion", "Politics", "Technology", "Sports", "Other"]
app = Flask(__name__)


@app.route("/")
def main():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def retrieve_category_for_tweet():
    tweet = request.form['tweet']
    cnn_category = cnn_classifier.predict(tweet)
    naive_bayes_category = nb.predict(tweet)
    knn_category = kNN.predict(tweet)
    return json.dumps(
        {"category-naive-bayes": categoryLabels[naive_bayes_category], "category-cnn": categoryLabels[cnn_category],
         "category-knn": categoryLabels[knn_category]})


if __name__ == "__main__":
    nb = naive_bayes_tweet_classifier()
    kNN = KNN_NLC_Classifer()
    cnn_classifier = cnn_classifier()
    app.run()
