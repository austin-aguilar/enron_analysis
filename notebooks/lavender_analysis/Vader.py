import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download

# Downloading VADER lexicon
download('vader_lexicon')

# sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Load DataFrame
data = pd.read_csv('/Users/tjra2/Downloads/processed_emails.csv')
df = pd.DataFrame(data)

# Function to analyze sentiment and determine label
def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        label = 'Positive'
    elif compound <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
    return pd.Series([compound, label])

# Apply the sentiment analysis to the 'text' column
df[['compound_score', 'sentiment_label']] = df['text'].apply(analyze_sentiment)

# Save the updated DataFrame to a CSV file
df.to_csv('vader_vader_sentiment_with_labels.csv', index=False)

# Filter negative data
negative_df = df[df['sentiment_label'] == 'Negative']
negative_df.to_csv('vader_negative_sentiment.csv', index=False)
print("Negative Sentiment Data:")

# Filter positive data
positive_df = df[df['sentiment_label'] == 'Positive']
positive_df.to_csv('vader_positive_sentiment.csv', index=False)
print("Positive Sentiment Data:")

# Filter neutral data
neutral_df = df[df['sentiment_label'] == 'Neutral']
neutral_df.to_csv('vader_neutral_sentiment.csv', index=False)
print("Neutral Sentiment Data:")

df.to_csv('vader_vader_sentiment_with_labels.csv', index=False)