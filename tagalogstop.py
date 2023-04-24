import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from stopwordsiso import stopwords as tagalog_stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import emoji
import string

# Load the dataset
df = pd.read_csv('datasetsApril.csv', encoding='ISO-8859-1')

# Drop duplicates in the 'Tweet' column
#df.drop_duplicates(subset='Tweet', inplace=True)

# Define a function to preprocess the text
def preprocess_text(text):

    if isinstance(text, float):
        return ""
    # Remove URLs, mentions, and non-alphanumeric characters
    text = re.sub(r'http\S+|www.\S+|@\S+|[^\w\s]', '', text.lower())
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode()
    # Split the text into words
    words = text.split()
    # Remove english and tagalog stop words and lemmatize the words
    lemmatizer = WordNetLemmatizer()
    english_stop_words = set(stopwords.words('english'))
    tagalog_stop_words = set(tagalog_stopwords(['tl']))
    stop_words = english_stop_words.union(tagalog_stop_words)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Join the words back into a string
    processed_text = " ".join(words)
    return processed_text

# Preprocess the text in the DataFrame
df['text'] = df['Tweet'].apply(lambda x: preprocess_text(x))



# Remove duplicate tweets
#df.drop_duplicates(subset='preprocessed_text', inplace=True)
df.to_csv('preprocess(April).csv', index=False)

# Show the preprocessed dataset
#print(df['preprocessed_text'])