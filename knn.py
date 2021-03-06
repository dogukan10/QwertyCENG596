from __future__ import division
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import genesis
nltk.download('genesis')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
genesis_ic = wn.ic(genesis, False, 0.0)

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import nltk
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from preprocessor import Preprocessor

class KNN_NLC_Classifer():
    def __init__(self, k=1, distance_type = 'path'):
        self.k = k
        self.distance_type = distance_type
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
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(tweets)

        tokenized_tweets = self.tokenizer.texts_to_sequences(tweets)
        num_tokens = [len(tokens) for tokens in tokenized_tweets]
        num_tokens = np.array(num_tokens)

        self.max_tokens = int(np.mean(num_tokens) + 2 * np.std(num_tokens))

        tokenized_tweets_padding = pad_sequences(tokenized_tweets, maxlen=self.max_tokens)
        X_train, X_test, y_train, y_test = train_test_split(tokenized_tweets_padding, labels, test_size=0.2,
                                                            random_state=123)
        self.fit(X_train, y_train)

    # This function is used for training
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    # This function runs the K(1) nearest neighbour algorithm and
    # returns the label with closest match. 
    def predict(self, x_test):
        x_test = [x_test]
        tokenized_tweets = self.tokenizer.texts_to_sequences(x_test)
        tokenized_tweets_padding = pad_sequences(tokenized_tweets, maxlen=self.max_tokens)

        self.x_test = tokenized_tweets_padding
        y_predict = []

        for i in range(len(self.x_test)):
            max_sim = 1000000
            max_index = 0
            for j in range(self.x_train.shape[0]):
                #temp = self.document_similarity(x_test[i], self.x_train[j])
                temp = abs(np.sum(self.x_test[i]) - np.sum(self.x_train[j]))
                if temp < max_sim:
                    max_sim = temp
                    max_index = j
            y_predict.append(self.y_train[max_index])
        return y_predict[0]

    def similarity_score(self, s1, s2, distance_type = 'path'):
          """
          Calculate the normalized similarity score of s1 onto s2
          For each synset in s1, finds the synset in s2 with the largest similarity value.
          Sum of all of the largest similarity values and normalize this value by dividing it by the
          number of largest similarity values found.

          Args:
              s1, s2: list of synsets from doc_to_synsets

          Returns:
              normalized similarity score of s1 onto s2
          """
          '''
          s1_largest_scores = []

          for i, s1_synset in enumerate(s1, 0):
              max_score = 0
              for s2_synset in s2:
                  if distance_type == 'path':
                      score = s1_synset.path_similarity(s2_synset, simulate_root = False)
                  else:
                      score = s1_synset.wup_similarity(s2_synset)                  
                      if score != None:
                          if score > max_score:
                              max_score = score
              if max_score != 0:
                  s1_largest_scores.append(max_score)
          
          mean_score = np.mean(s1_largest_scores)
          '''
          return mean_score

    def document_similarity(self,doc1, doc2):
          """Finds the symmetrical similarity between doc1 and doc2"""

          synsets1 = self.doc_to_synsets(doc1)
          synsets2 = self.doc_to_synsets(doc2)
          
          #return (self.similarity_score(synsets1, synsets2) + self.similarity_score(synsets2, synsets1)) / 2
          return (abs(synsets1 - synsets2))
    def doc_to_synsets(self, doc):
        """
            Returns a list of synsets in document.
            Tokenizes and tags the words in the document doc.
            Then finds the first synset for each word/tag combination.
        If a synset is not found for that combination it is skipped.

        Args:
            doc: string to be converted

        Returns:
            list of synsets
        """
        tokens = word_tokenize(str(doc) + ' ')
        
        l = []
        tags = nltk.pos_tag([tokens[0] + ' ']) if len(tokens) == 1 else nltk.pos_tag(tokens)
        
        for token, tag in zip(tokens, tags):
            syntag = self.convert_tag(tag[1])
            syns = wn.synsets(token, syntag)
            if (len(syns) > 0):
                l.append(syns[0])
        return l    
        
    def convert_tag(self, tag):
        """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
        tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
        try:
            return tag_dict[tag[0]]
        except KeyError:
            return None

def accuracy(y_true, y_pred):
    accuracy = np.count_nonzero(np.asarray(y_true,dtype=np.float32)==np.asarray(y_pred,dtype=np.float32)) / len(y_true)
    return accuracy
	
# print("Accuracy", accuracy(y_test, predictions))
