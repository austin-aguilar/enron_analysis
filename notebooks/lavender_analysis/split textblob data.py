import pandas as pd

# Load the Enron processed data
data = pd.read_csv('textblob_sorted.csv')
df = pd.DataFrame(data)

# Filtered DataFrame with Sentiment <= -0.05
negative_df = df[df['sentiment'] <= -0.05]

# Save filtered negative sentiment
negative_df.to_csv('texblob_negative_df.csv', index=False)

print('saved negative textblob')