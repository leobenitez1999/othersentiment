import pandas as pd
import json
from collections import defaultdict

# Load data from CSV file
data = pd.read_csv("filtered_processed - Balanced copy.csv")

# Group data by date and sentiment label
grouped_data = data.groupby(["Date", "Sentiment_Labels"]).count()["Tweet"].reset_index()

# Create dictionary to hold sentiment counts for each date
sentiment_counts = defaultdict(dict)

# Populate dictionary with sentiment counts
for _, row in grouped_data.iterrows():
    sentiment_counts[row["Date"]][row["Sentiment_Labels"]] = row["Tweet"]

# Convert dictionary to lists for Chart.js input
labels = sorted(sentiment_counts.keys())
positive_counts = [sentiment_counts[date].get("positive", 0) for date in labels]
negative_counts = [sentiment_counts[date].get("negative", 0) for date in labels]

# Create JSON data for Chart.js
chart_data = {
    "type": "line",
    "data": {
        "labels": labels,
        "datasets": [
            {
                "label": "Positive",
                "data": positive_counts,
                "fill": False,
                "borderColor": "green"
            },
            {
                "label": "Negative",
                "data": negative_counts,
                "fill": False,
                "borderColor": "red"
            }
        ]
    },
    "options": {
        "title": {
            "display": True,
            "text": "Sentiment Trend Graph"
        }
    }
}

# Convert JSON data to string
chart_data_str = json.dumps(chart_data)

# Output Chart.js code to HTML file
with open("sentiment_trend.html", "w") as f:
    f.write("""
        <html>
            <head>
                <title>Sentiment Trend Graph</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            </head>
            <body>
                <canvas id="chart"></canvas>
                <script>
                    var chartData = %s;
                    var ctx = document.getElementById('chart').getContext('2d');
                    var chart = new Chart(ctx, chartData);
                </script>
            </body>
        </html>
    """ % chart_data_str)
