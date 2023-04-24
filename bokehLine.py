import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from io import BytesIO
import base64

# Read CSV file
df = pd.read_csv('filtered_processed - Copy.csv')

# Convert Date column to datetime format
df['Datetime'] = pd.to_datetime(df['Date'])

# Create a pivot table to count number of tweets by date and sentiment
pivot = pd.pivot_table(df, index='Datetime', columns='Sentiment_Labels', values='Tweet', aggfunc='count')
pivot.fillna(0, inplace=True)

# Create a figure object
fig, ax = plt.subplots(figsize=(10, 5))

# Create a colormap
colors = np.linspace(0, 1, len(pivot.index))
cmap = plt.cm.Greens

# Plot positive sentiment as green dots
ax.scatter(pivot.index, pivot['positive'], c=colors, cmap=cmap, s=200, alpha=0.8, label='Positive')

# Plot negative sentiment as red dots
ax.scatter(pivot.index, pivot['negative'], c=colors, cmap=cmap, s=200, alpha=0.8, label='Negative')

# Add legend and axis labels
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Number of Tweets')
ax.set_title('Sentiment Trend by Date')

# Add gridlines
ax.grid(True)

# Save the plot as a PNG image
sentiment_trend_image = BytesIO()
fig.savefig(sentiment_trend_image, format='png')
sentiment_trend_image.seek(0)
sentiment_trend_image_url = base64.b64encode(sentiment_trend_image.getvalue()).decode()
with open('sentiment_trend.png', 'wb') as f:
    f.write(base64.b64decode(sentiment_trend_image_url))
