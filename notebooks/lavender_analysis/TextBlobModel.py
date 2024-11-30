import pandas as pd
from textblob import TextBlob

# Load the Enron processed data
data = pd.read_csv('/Users/tjra2/Downloads/processed_emails.csv')
df = pd.DataFrame(data)

#Applyng TextBlob to the dataframe and setting sentiment and subjectivity results
df['sentiment']=df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Filter rows with no sentiment calculated (reduces output lines with no text data)
filtered_df = df[df['sentiment'].notna()]

# Output the results
filtered_df.to_csv('textblob_filtered_full.csv', index=False)


