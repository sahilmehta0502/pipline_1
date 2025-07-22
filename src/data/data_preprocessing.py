import os
import re
import numpy as np
import pandas as pd
import nltk
import string
import logging
from typing import Any
from pandas import DataFrame
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(
    filename="logs/data_preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def lemmatization(text: str) -> str:
    """Lemmatize each word in the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text: str) -> str:
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text: str) -> str:
    """Remove all digits from the text."""
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text: str) -> str:
    """Convert all words in the text to lowercase."""
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def removing_punctuations(text: str) -> str:
    """Remove punctuations and extra whitespace from the text."""
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "", )
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def removing_urls(text: str) -> str:
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df: DataFrame) -> None:
    """Set text to NaN if sentence has fewer than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df: DataFrame) -> DataFrame:
    """Apply all preprocessing steps to the 'content' column of the DataFrame."""
    try:
        df.content = df.content.apply(lambda content: lower_case(content))
        df.content = df.content.apply(lambda content: remove_stop_words(content))
        df.content = df.content.apply(lambda content: removing_numbers(content))
        df.content = df.content.apply(lambda content: removing_punctuations(content))
        df.content = df.content.apply(lambda content: removing_urls(content))
        df.content = df.content.apply(lambda content: lemmatization(content))
        logging.info("Text normalization completed successfully.")
        return df
    except Exception as e:
        logging.error("Error during text normalization: %s", e)
        raise

def normalized_sentence(sentence: str) -> str:
    """Apply all preprocessing steps to a single sentence."""
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error("Error normalizing sentence: %s", e)
        raise

def load_data(path: str) -> DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(path)
        logging.info("Data loaded from %s", path)
        return df
    except Exception as e:
        logging.error("Error loading data from %s: %s", path, e)
        raise

def save_data(df: DataFrame, path: str) -> None:
    """Save DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info("Data saved to %s", path)
    except Exception as e:
        logging.error("Error saving data to %s: %s", path, e)
        raise

def main() -> None:
    """Main function to orchestrate data preprocessing."""
    try:
        train_data = load_data("data/raw/train.csv")
        test_data = load_data("data/raw/test.csv")
        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)
        save_data(train_data, "data/processed/train.csv")
        save_data(test_data, "data/processed/test.csv")
        logging.info("Data preprocessing pipeline completed successfully.")
    except Exception as e:
        logging.critical("Data preprocessing pipeline failed: %s", e)
        raise

if __name__ == "__main__":
    main()