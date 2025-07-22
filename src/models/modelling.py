import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from typing import Tuple

# Configure logging
logging.basicConfig(
    filename="logs/modelling.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_params(params_path: str) -> dict:
    """Load model parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully from %s", params_path)
        return params
    except Exception as e:
        logging.error("Error loading parameters: %s", e)
        raise

def load_train_data(train_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data and separate features and labels."""
    try:
        train_data = pd.read_csv(train_path)
        X_train = train_data.drop(columns=['label']).values
        y_train = train_data['label'].values
        logging.info("Training data loaded from %s", train_path)
        return X_train, y_train
    except Exception as e:
        logging.error("Error loading training data: %s", e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    """Train a RandomForestClassifier model."""
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error("Error training model: %s", e)
        raise

def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """Save the trained model to a file."""
    try:
        pickle.dump(model, open(model_path, "wb"))
        logging.info("Model saved to %s", model_path)
    except Exception as e:
        logging.error("Error saving model: %s", e)
        raise

def main() -> None:
    """Main function to orchestrate model training and saving."""
    try:
        params = load_params("params.yaml")
        n_estimators = params['modelling']['n_estimators']
        max_depth = params['modelling']['max_depth']
        X_train, y_train = load_train_data("data/interim/train.csv")
        model = train_model(X_train, y_train, n_estimators, max_depth)
        save_model(model, "models/random_forest_model.pkl")
        print("Model trained and saved successfully.")
        logging.info("Model training pipeline completed successfully.")
    except Exception as e:
        logging.critical("Model training pipeline failed: %s", e)
        raise

if __name__ == "__main__":
    main()