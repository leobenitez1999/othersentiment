import snscrape.modules.twitter as sntwitter
import pandas as pd


query = "(#ewallet) until:2020-01-01 since:2010-01-01"
tweets = []
limit = 5
languages = ["en", "tl"]

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) == limit:
        break
    elif tweet.lang in languages:
        tweets.append([tweet.date, tweet.content])

df = pd.DataFrame(tweets, columns=['Date', 'Tweet'])
print(df)
