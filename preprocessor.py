import re
from nltk.corpus import stopwords


class Preprocessor:

    # initialize Preprocessor with tweets and PorterStemmer
    def __init__(self, tweets, ps):
        self.tweets = tweets
        self.ps = ps

    # start preprocessing
    def start(self):
        for i, k in enumerate(self.tweets):
            self.tweets[i] = " ".join(self.split_into_stem(k)).split()

        return self.remove_stop_words(self.tweets)

    def split_into_stem(self, message):
        return [self.remove_numeric(self.strip_emoji(self.single_character_remove(self.remove_punctuation
                                                                                  (self.remove_hyperlinks
                                                                                   (self.remove_hashtags
                                                                                    (self.remove_username
                                                                                     (self.stem_word(word)))))))) for
                word in
                message.split()]

    # stem the word
    def stem_word(self, word):
        return self.ps.stem(word)

    # Static Methods

    # Remove username
    @staticmethod
    def remove_username(tweet):
        return re.sub('@[^\s]+', '', tweet)

    # Remove hashtag
    @staticmethod
    def remove_hashtags(tweet):
        return re.sub(r'#[^\s]+', '', tweet)

    # Remove link
    @staticmethod
    def remove_hyperlinks(tweet):
        return re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)

    # Remove numeric character
    @staticmethod
    def remove_numeric(word):
        char_list = [char for char in word if not char.isdigit()]
        return "".join(char_list)

    # Remove punctuation
    @staticmethod
    def remove_punctuation(tweet):
        return re.sub(r'[^\w\s]', '', tweet)

    # Remove single character
    @staticmethod
    def single_character_remove(tweet):
        return re.sub(r'(?:^| )\w(?:$| )', ' ', tweet)

    # Remove emoji
    @staticmethod
    def strip_emoji(text):
        RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
        return RE_EMOJI.sub(r'', text)

    @staticmethod
    def remove_stop_words(tweet_list):
        filtered_words = []

        stop_words = stopwords.words('english')

        for i in tweet_list:
            filtered_sentence = [w for w in i if not w in stop_words]
            filtered_words.append(" ".join(filtered_sentence))

        return filtered_words
