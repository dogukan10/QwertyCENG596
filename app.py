from flask import Flask, render_template, request, json

app = Flask(__name__)


@app.route("/")
def main():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def retrieve_category_for_tweet():
    tweet = request.form['tweet']
    return json.dumps({"category": "<CATEGORY_HERE>"})


if __name__ == "__main__":
    app.run()
