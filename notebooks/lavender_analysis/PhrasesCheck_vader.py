import re
import pandas as pd

# Function to search text for fraud phrases from keyword list
# Searches with case ignored; returns list of matches
def search_fraud_phrases(text, fraud_phrases):
    matches = []
    for phrase in fraud_phrases:
        if re.search(re.escape(phrase), text, re.IGNORECASE):
            matches.append(phrase)
    return matches

# Load phrases from CSV
fraud_phrases = pd.read_csv('phrases.csv')

# Sets phrase column as a list
phrases_list = fraud_phrases['Phrase'].dropna().tolist()

# Text to check from negative sentiment file
neg_text = pd.read_csv('vader_negative_sentiment.csv')
df_neg_text = pd.DataFrame(neg_text)


# Search for matches using apply function for dataframe
df_neg_text['matches'] = df_neg_text['text'].apply(lambda text:search_fraud_phrases(text, phrases_list))

# Output the results
df_text_with_matches = df_neg_text[df_neg_text['matches'].apply(len) > 0]
df_text_with_matches.to_csv('results/vader_text_with_matches_neg.csv', index=False)
print('all done')