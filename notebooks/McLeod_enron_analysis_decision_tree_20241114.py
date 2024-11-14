# -*- coding: utf-8 -*-
"""
Enron Analysis - Decision Tree
 
"""

import pandas as pd

# Define the file path
file_path = r"G:\Academics\Georgia Tech - MS CS\INTA 6450 - Data Analytics & Security\Final Project\enron-analysis-main\enron_analysis\data\data\processed_emails_top_2mm_lines.csv"

# Pandas Table Display Options
#pd.set_option('display.max_rows', 10)  # No row limit
#pd.set_option('display.max_columns', None)  # No column limit

#%% 
"""
# Import processed files
Columns include:
- Time of Day
- End of fiscal period?
- Contains Wrongdoing Keywords?
- Keyword Density
- Outside work hours?
"""

#%% 
# import
import pandas as pd
from datetime import datetime

#from google.colab import files # Importing the 'files' object from google.colab
#files.upload()


#%% 
# Read the File
main = pd.read_csv(file_path)
main.head()
main.columns
# main.replace(to_replace='NaN', value=None, inplace=True)
# main.drop(['message', 'file'], axis = 1, inplace=True)


#%%
# Convert 'date' column to datetime
main['date'] = pd.to_datetime(main['date'], errors='coerce')


# Check for rows where 'date' conversion was unsuccessful
print("Rows with invalid date format:", main[main['date'].isna()])

#%%
# Extract hour from the 'date' column
#main['Hour'] = main['date'].dt.hour
main['Hour'] = main['date'].apply(lambda x: x.hour if pd.notnull(x) else None)
print(main['Hour'])

#%%
print("File Column:"); print(main['file'].head())

print("\nMessage Column:"); print(main['message'].head())

print("\nText Column:"); print(main['text'].head())

print("\nSender Column:"); print(main['sender'].head())

print("\nRecipient1 Column:"); print(main['recipient1'].head())

print("\nRecipient2 Column:"); print(main['recipient2'].head())

print("\nRecipient3 Column:"); print(main['recipient3'].head())

print("\nSubject Column:"); print(main['Subject'].head())

print("\nFolder Column:"); print(main['folder'].head())

print("\nDate Column:"); print(main['date'].head())

print("\nHour Column:"); print(main['Hour'].head())

#%%

# Assuming 'main' DataFrame is already loaded as in the previous code

# Function to categorize time of day
def categorize_time(hour):
  if 0 <= hour < 6:
    return "Late Night"
  elif 6 <= hour < 12:
    return "Morning"
  elif 12 <= hour < 18:
    return "Afternoon"
  else:
    return "Evening"


#%%

# Create 'Time of Day' column
main['Time of Day'] = main['Hour'].apply(categorize_time)

# Example of flagging outside work hours (adjust as needed)
main['Outside work hours?'] = (main['Hour'] < 9) | (main['Hour'] > 17)

# Example of flagging end of fiscal period (replace with your actual logic)
# Enron fiscal period ends on December 31st
main['End of fiscal period?'] = main['date'].apply(lambda x: x.month == 12 and x.day == 30)

#%%
# Keywords
keywords = [
    "between us", "work around this", "X", "cover up", "illegal", "fraud", "embezzle", "bribe", "kickback", 
    "misappropriation", "under the table", "unreported", "deceptive", "money laundering", "collusion", "scheme", 
    "insider trading", "conflict of interest", "conspiracy", "unethical", "off the books","off book", "false documentation", 
    "tampering", "conceal", "mislead", "scam", "shady", "falsify", "extortion", "abuse of power", "negligence", 
    "coercion", "illegal activity", "misstatement", "secret agreement", "unlawful", "accounting fraud", 
    "unreported income", "double-dealing", "bribery", "hush money", "kickback scheme", "insider info", "phoney", 
    "illicit", "financial manipulation", "unethical practice", "whistleblower", "dodgy deal","urgent"
]


# Check if any of the specified keywords are present in the relevant column
# The 'str.contains()' method looks for any of the keywords in the text of each row. Ignores case
# The '|' (pipe) character is used to join the keywords, so it's treated as an OR condition for matching
main["Contains Wrongdoing Keywords?"] = main['text'].str.contains('|'.join(keywords), case=False, na=False) 

# Calculate the keyword density, i.e., the number of wrongdoing keywords in the relevant column
# This splits each row's text into individual words, then checks if each word is in the list of keywords
# The 'sum' function counts how many keywords appear in the split text (if it's a valid list of words)
main['Keyword Density'] = main['text'].str.split().apply(lambda x: sum(word in keywords for word in x if isinstance(x, list)))  


#%%
"""
# User Sentiment library
from textblob import TextBlob

# Function to classify sentiment based on polarity
def classify_sentiment_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    # Classify based on polarity
    if polarity > 0.1:
        return "Urgent"  # Positive polarity indicates urgency (adjust threshold as needed)
    elif polarity < -0.1:
        return "Tense"  # Negative polarity indicates tense/negative sentiment
    else:
        return "Neutral"  # Neutral polarity indicates neutral sentiment

# Apply the classification function to each row in the 'text' column
main['Sentiment'] = main['text'].apply(classify_sentiment_textblob)

# Check the result
print(main[['text', 'Sentiment']].head())
"""


#%%

# Print all columns in the head of the dataframe
print(main.head())

# Optionally, print the column names as well
print("Column names:", main.columns.tolist())

#%%
try:
    from textblob import TextBlob
    print("textblob is installed")
except ImportError:
    print("textblob is not installed")

## No idea why not, it's installed.

#######
# END OF PREPROCESSING
#######


#%%
"""
-- Create a decision tree based on the values: 

Time of Day
Outside work hours
End of fiscal period 
Contains WrongdoingKeywords
Keyword Density 

"""

####### DECISION TREE ####### 


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

#%% 

try:
    from sklearn.model_selection import train_test_split    
    print("sklearn is installed")
except ImportError:
    print("sklearn is not installed")




#%%
# Preprocessing for categorical columns (e.g., "Time of Day")
# You can convert "Time of Day" to numerical values like:
time_of_day_map = {
    "Late Night": 0,
    "Morning": 1,
    "Afternoon": 2,
    "Evening": 3
}
main['Time of Day'] = main['Time of Day'].map(time_of_day_map)


#%% 
# Input X columns. Our indpendent variables.
inputs = ['Time of Day', 'End of fiscal period?', 'Contains Wrongdoing Keywords?', 'Keyword Density']
# 'Outside work hours?'

# Target is the column you want to predict. # WIP - we have not coded which emails were fraudulent. Will use 1 for wrongdoing, 0 for no wrongdoing.
target = 'Outside work hours?'  # Temporary Y variable until we code for wrongdoing. Should be 100% accurate.


# Step 1: Prepare feature and target data
X = main[inputs]  # Feature columns
y = main[target]  # Target column

# Step 2: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 4: Predict the target on the test set
y_pred = clf.predict(X_test)

# Step 5: Evaluate the performance of the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualization of the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=inputs, class_names=['Not Flagged', 'Flagged'], filled=True, fontsize=10)
plt.title("Decision Tree for Flagged Emails Prediction")
plt.show()







#%%
# Export this as a CSV

