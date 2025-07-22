import numpy as np
import pandas as pd
import os
import yaml
import logging
from typing import Tuple
from pandas import DataFrame

from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    filename="logs/data_ingestion.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

pd.set_option('future.no_silent_downcasting', True)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully from %s", params_path)
        return params
    except Exception as e:
        logging.error("Error loading parameters: %s", e)
        raise

def fetch_data(url: str) -> DataFrame:
    """Fetch dataset from a given URL."""
    try:
        df = pd.read_csv(url)
        logging.info("Data loaded successfully from %s", url)
        return df
    except Exception as e:
        logging.error("Error fetching data: %s", e)
        raise

def preprocess_data(df: DataFrame) -> DataFrame:
    """Preprocess the dataset by dropping unnecessary columns and filtering sentiments."""
    try:
        df = df.drop(columns=['tweet_id'])
        filtered_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        filtered_df['sentiment'] = filtered_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logging.info("Data preprocessing completed successfully.")
        return filtered_df
    except Exception as e:
        logging.error("Error during preprocessing: %s", e)
        raise

def split_data(df: DataFrame, test_size: float) -> Tuple[DataFrame, DataFrame]:
    """Split the data into training and testing sets."""
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        logging.info("Data split into train and test sets.")
        return train_data, test_data
    except Exception as e:
        logging.error("Error splitting data: %s", e)
        raise

def save_data(train_data: DataFrame, test_data: DataFrame, train_path: str, test_path: str) -> None:
    """Save train and test data to CSV files."""
    try:
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info("Train and test data saved to %s and %s", train_path, test_path)
    except Exception as e:
        logging.error("Error saving data: %s", e)
        raise

def main() -> None:
    """Main function to orchestrate data ingestion."""
    try:
        params = load_params("params.yaml")
        test_size: float = params['data_ingestion']['test_size']
        url: str = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        df = fetch_data(url)
        final_df = preprocess_data(df)
        train_data, test_data = split_data(final_df, test_size)
        save_data(train_data, test_data, "data/raw/train.csv", "data/raw/test.csv")
        logging.info("Data ingestion pipeline completed successfully.")
    except Exception as e:
        logging.critical("Data ingestion pipeline failed: %s", e)
        raise

if __name__ == "__main__":
    main()