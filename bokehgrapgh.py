import pandas as pd
from datetime import datetime as dt
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

# Read CSV file
df = pd.read_csv('filtered_processed - Copy.csv')

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S%z')

# Group by Date and Sentiment Label and count number of tweets
grouped = df.groupby(['Date', 'Sentiment_Labels']).count().reset_index()

# Clean up Sentiment_Labels column
grouped['Sentiment_Labels'] = grouped['Sentiment_Labels'].str.lower().str.strip().replace({"positive ": "positive", "negative ": "negative"})

# Filter out any neutral labels (optional)
grouped = grouped[grouped['Sentiment_Labels'].isin(['positive', 'negative'])]

# Assign color based on sentiment label
grouped['color'] = ['skyblue' if label == 'positive' else 'lightpink' for label in grouped['Sentiment_Labels']]

# Create a ColumnDataSource object
source = ColumnDataSource(grouped)

# Create a figure object
p = figure(x_axis_type='datetime', width=600, height=400, title='Sentiment Trend')

# Add circle glyphs for each sentiment label
p.circle(x='Date', y='Tweet', source=source, size=10, color='color', alpha=0.5, legend_group='Sentiment_Labels')

# Add legend
p.legend.location = 'top_left'

# Show the figure
show(p)
