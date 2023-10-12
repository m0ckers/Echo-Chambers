import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
# Initialize NLTK (if needed)
# nltk.download('punkt')

def split_tweets(df):
#Create new dataframe splitting the tweets
    new_rows = []
    # Iterate through each row in the original DataFrame
    for index, row in df.iterrows():
        segments = row['tweets'].split(" | ")
        for segment in segments:
            new_row = row.copy()  # Create a copy of the original row
            new_row['tweets'] = segment  # Replace the 'tweets' column with the segment
            new_rows.append(new_row)  # Append the new row to the list
    # Create a new DataFrame from the list of split rows
    split_df = pd.DataFrame(new_rows)
    #Display the resulting split DataFrame
    df = split_df
    df = df.reset_index(drop=True)
    return(df)

def preprocess_tweet(tweet, min_length=4, max_length=10):
    # Remove URLs, mentions, special characters, numbers, and stuff between ()
    tweet = re.sub(r"http\S+|www\S+|https\S+|@\w+|[^a-zA-Z']|\(.*?\)", " ", tweet, flags=re.MULTILINE)
    # Tokenize the tweet
    tokens = word_tokenize(tweet)
    # Convert to lowercase
    tokens = [token.lower() for token in tokens]
    # Remove stopwords (uncomment if needed)
    # stop_words = set(stopwords.words("english"))
    # tokens = [token for token in tokens if token not in stop_words]
    # Join the tokens back into a single string
    preprocessed_tweet = " ".join(tokens)
    return preprocessed_tweet

def text_processor():
    text_processor = TextPreProcessor(
        segmenter="twitter",
        corrector="twitter",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize)
    return text_processor

def preprocess(df):
    preprocessed_texts = []
    text_processor_instance = text_processor()

    for text in df['tweets']:
        preprocessed_text = ' '.join(text_processor_instance.pre_process_doc(text))
        preprocessed_text = preprocess_tweet(preprocessed_text)
        preprocessed_texts.append(preprocessed_text)

    labels = list(df['stance'])
    labels = [2 if label == -1 else label for label in labels]
    df['stance'] = labels
    df['Preprocessed'] = preprocessed_texts
    #Labels Text
    lt = []
    for label in df['Labels']:
        sentiment = 'Positive' if label == 2 else ('Neutral' if label == 0 else 'Negative')
        lt.append(sentiment)
    df['Labels Text'] = lt
    df = df[['user','tweets','Preprocessed', 'Labels', 'Labels Text']]
    df = df.reset_index(drop=True)
    return df

def split(df):# On pre-processed tweets
    from sklearn.model_selection import train_test_split
    labels = df['Stance']
    train_text, test_text, train_labels, test_labels = train_test_split(df['Preprocessed'], df['stance'], random_state=2018,
                                                                        test_size=0.2,
                                                                        stratify=labels)
    train_labels = train_labels.tolist()
    test_labels = test_labels.tolist()
    # Get the lists of sentences and their labels.
    print(len(train_labels), len(train_text))
    return(train_text, train_labels)