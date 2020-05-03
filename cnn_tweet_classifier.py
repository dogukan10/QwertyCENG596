import nltk
import numpy as np
from keras.layers import Conv1D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from  keras.utils import plot_model
import matplotlib.pyplot as plt
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
num_epochs = 20
num_class = 8

# reshape the input since Conv1D expects three dimensional inputs
tokenized_tweets_padding = tokenized_tweets_padding.reshape(tokenized_tweets_padding.shape[0],
                                                            tokenized_tweets_padding.shape[1], 1)

# Create CNN model
model = Sequential()
model.add(Conv1D(32, 3, input_shape=(tokenized_tweets_padding.shape[1], tokenized_tweets_padding.shape[2]),
                 activation='relu'))
model.add(Flatten())
model.add(Dense(num_class, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=sgd,
              metrics=['sparse_categorical_accuracy'])

model.summary()
# run model
history = model.fit(np.array(tokenized_tweets_padding), np.array(labels), batch_size=batch_size, epochs=num_epochs,
          validation_split=0.1,
          shuffle=True, verbose=2)

# Plot training & validation accuracy values
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# save model weights for the future
model.save_weights('cnn.h5')

# Plot the CNN model
plot_model(model,to_file="model.png",show_layer_names=True,show_shapes=True)
