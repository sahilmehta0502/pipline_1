import numpy as np
import pandas as pd
import logging
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple

# Configure logging
logging.basicConfig(
    filename="logs/feature_engg.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed training and test data, dropping rows with missing 'content'."""
    try:
        train_data = pd.read_csv(train_path).dropna(subset=['content'])
        test_data = pd.read_csv(test_path).dropna(subset=['content'])
        logging.info("Train and test data loaded successfully.")
        return train_data, test_data
    except Exception as e:
        logging.error("Error loading train/test data: %s", e)
        raise

def extract_features_and_labels(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract features and labels from train and test datasets."""
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        logging.info("Features and labels extracted successfully.")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error("Error extracting features/labels: %s", e)
        raise

def vectorize_data(X_train: np.ndarray, X_test: np.ndarray, max_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit Bag-of-Words vectorizer on training data and transform both train and test data."""
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logging.info("Data vectorization completed successfully.")
        return X_train_bow, X_test_bow
    except Exception as e:
        logging.error("Error during vectorization: %s", e)
        raise

def save_features(train_bow, y_train, test_bow, y_test, train_out: str, test_out: str) -> None:
    """Save transformed features and labels to interim CSV files."""
    try:
        train_df = pd.DataFrame(train_bow.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(test_bow.toarray())
        test_df['label'] = y_test
        train_df.to_csv(train_out, index=False)
        test_df.to_csv(test_out, index=False)
        logging.info("Transformed features and labels saved to %s and %s", train_out, test_out)
    except Exception as e:
        logging.error("Error saving features: %s", e)
        raise

def main() -> None:
    """Main function to orchestrate feature engineering."""
    try:
        params = load_params("params.yaml")
        max_features = params['feature_engg']['max_features']
        train_data, test_data = load_data("data/processed/train.csv", "data/processed/test.csv")
        X_train, y_train, X_test, y_test = extract_features_and_labels(train_data, test_data)
        X_train_bow, X_test_bow = vectorize_data(X_train, X_test, max_features)
        save_features(X_train_bow, y_train, X_test_bow, y_test, "data/interim/train.csv", "data/interim/test_bow.csv")
        logging.info("Feature engineering pipeline completed successfully.")
    except Exception as e:
        logging.critical("Feature engineering pipeline failed: %s", e)
        raise

if __name__ == "__main__":
    main()