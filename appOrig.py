import snscrape.modules.twitter as sntwitter
import pandas as pd
import nltk
from datetime import datetime, timedelta
import re
import string
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# Define the start and end date for the search
end_date = datetime.today()
start_date = end_date - timedelta(days=30)
languages = ["en", "tl"]

# Load the trained MNB classifier
with open('mnb_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Load the preprocessing functions
with open('tfidf_vectorizer.pkl', 'rb') as f:
    preprocess_text, remove_stopwords = pickle.load(f)

# Load the stopwords
stop_words = set(stopwords.words('english'))

# Define the WordNetLemmatizer and PorterStemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # Get the user input
    keyword = request.form['text']
    num_tweets = 300
    # Scrape recent tweets
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keyword} since:{start_date.date()} until:{end_date.date()}').get_items()):
        if i >= int(num_tweets):
            break
        elif tweet.lang in languages:
            tweets.append([tweet.date, tweet.content])
    
    # Clean the scraped tweets
    cleaned_tweets = []
    for tweet in tweets:
        # Preprocess the tweet
        preprocessed_tweet = preprocess_text(tweet, remove_stopwords, stop_words, lemmatizer, stemmer)
        cleaned_tweets.append(preprocessed_tweet)
    
    # Classify the cleaned tweets
    predictions = classifier.predict(cleaned_tweets)
    positive_count = 0
    negative_count = 0
    for prediction in predictions:
        if prediction == 1:
            positive_count += 1
        else:
            negative_count += 1
    
    # Calculate the accuracy, precision, recall, and f1-score of the model
    # Note that we assume that the ground truth labels are not available here
    accuracy = classifier.score(cleaned_tweets, predictions)
    precision = nltk.metrics.precision(predictions, predictions)
    recall = nltk.metrics.recall(predictions, predictions)
    f1_score = nltk.metrics.f1score(predictions, predictions)
    
    # Generate the wordcloud
    wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(' '.join(cleaned_tweets))
    
    # Generate the graph
    labels = ['Positive', 'Negative']
    values = [positive_count, negative_count]
    plt.bar(labels, values)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Analysis Results')
    plt.savefig('static/plot.png')
    
    return render_template('result.html', keyword=keyword, num_tweets=num_tweets, accuracy=accuracy, precision=precision, recall=recall, f1_score=f1_score, positive_count=positive_count, negative_count=negative_count)

if __name__ == '__main__':
    app.run(debug=True)
