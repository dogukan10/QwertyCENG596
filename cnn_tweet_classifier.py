import numpy as np
from keras import layers
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import nltk
from preprocessor import Preprocessor

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

# CNN architecture

# Training params
batch_size = 256
num_epochs = 10

input_dim = tokenized_tweets_padding.shape[1]

# Create CNN model
model = Sequential()

model.add(layers.Dense(25, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(8, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# run model
model.fit(np.array(tokenized_tweets_padding), np.array(labels), batch_size=batch_size, epochs=num_epochs,
          validation_split=0.1,
          shuffle=True, verbose=2)
# save model weights for the future
model.save_weights('cnn.h5')
