import numpy as np
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import os
from nltk.stem import SnowballStemmer

# Load processed training and test data, dropping rows with missing 'content'
train_data = pd.read_csv("data/processed/train.csv").dropna(subset=['content'])
test_data = pd.read_csv("data/processed/test.csv").dropna(subset=['content'])

# Extract features and labels from train and test datasets
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Initialize Bag-of-Words vectorizer
vectorizer = CountVectorizer()

# Fit vectorizer on training data and transform both train and test data
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Convert sparse matrices to DataFrames and add label columns
train_df = pd.DataFrame(X_train_bow.toarray())
train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())
test_df['label'] = y_test   

# Save transformed features and labels to interim CSV files
train_df.to_csv("data/interim/train.csv", index=False)
test_df.to_csv("data/interim/test_bow.csv", index=False)