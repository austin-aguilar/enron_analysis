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
fraud_phrases = pd.read_csv('phrases2.csv')

# Sets phrase column as a list
phrases_list = fraud_phrases['Phrase'].dropna().tolist()

# Text to check from negative sentiment file
textblob_full = pd.read_csv('textblob_filtered_full.csv')
df_full_text = pd.DataFrame(textblob_full)

# Search for matches using apply function for dataframe
df_full_text['matches'] = df_full_text['text'].apply(lambda text:search_fraud_phrases(text, phrases_list))

# Output the results
df_text_with_matches = df_full_text[df_full_text['matches'].apply(len) > 0]
df_text_with_matches.to_csv('results/textblob_text_with_matches_full.csv', index=False)
print('all done')