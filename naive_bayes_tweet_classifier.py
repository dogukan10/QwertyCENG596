from sklearn.model_selection import train_test_split
import nltk
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from preprocessor import Preprocessor
from naive_bayes import NaiveBayes


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# download stopwords
nltk.download('stopwords')

# tweets and their labels
tweets = []
labels = []

# retrieve tweets
file = open("tweets.txt", "r")

# tweet to be added to the tweets list
tweet = ""
# line to be read from the file
line = file.readline()
while line:
    #  if the line is the label, add tweet and its label to the corresponding lists
    if line.startswith("$$$$$"):
        # add tweet
        tweets.append(tweet)
        # add label
        labels.append(int(line[5:].replace("\n", "")))
        # clear tweet object
        tweet = ""
    #  else, the line is a part of the tweet
    else:
        tweet += line.replace("\n", "").strip().lower()
    # read new line
    line = file.readline()

# Preprocessing
preprocessor = Preprocessor(tweets, nltk.PorterStemmer())
tweets = preprocessor.start()

# Tokenize tweets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)

tokenized_tweets = tokenizer.texts_to_sequences(tweets)
num_tokens = [len(tokens) for tokens in tokenized_tweets]
num_tokens = np.array(num_tokens)

max_tokens = int(np.mean(num_tokens) + 2 * np.std(num_tokens))

tokenized_tweets_padding = pad_sequences(tokenized_tweets, maxlen=max_tokens)
X_train, X_test, y_train, y_test = train_test_split(tokenized_tweets_padding, labels, test_size=0.2, random_state=123)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print("Accuracy", accuracy(y_test, predictions))
