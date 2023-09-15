import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_tweet(tweet, min_length=4, max_length=10):
    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    # Remove mentions
    tweet = re.sub(r"@\w+", "", tweet)
    # Remove special characters and numbers
    tweet = re.sub(r"[^a-zA-Z']", " ", tweet)
    #Remove stuff between ()
    tweet = re.sub(r"\(.*?\)", "", tweet)
    # Tokenize the tweet
    tokens = word_tokenize(tweet)
    # Convert to lowercase
    tokens = [token.lower() for token in tokens]
    # Remove stopwords
    #stop_words = set(stopwords.words("english"))
    #tokens = [token for token in tokens if token not in stop_words]
    # Join the tokens back into a single string
    preprocessed_tweet = " ".join(tokens)

    return preprocessed_tweet