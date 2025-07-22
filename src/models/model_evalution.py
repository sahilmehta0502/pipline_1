import pandas as pd
import pickle
import json
import logging
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Configure logging
logging.basicConfig(
    filename="logs/model_evalution.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_model(model_path: str) -> Any:
    """Load a trained model from file."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info("Model loaded from %s", model_path)
        return model
    except Exception as e:
        logging.error("Error loading model from %s: %s", model_path, e)
        raise

def load_test_data(test_path: str) -> pd.DataFrame:
    """Load test data from CSV file."""
    try:
        df = pd.read_csv(test_path)
        logging.info("Test data loaded from %s", test_path)
        return df
    except Exception as e:
        logging.error("Error loading test data from %s: %s", test_path, e)
        raise

def evaluate_model(model: Any, test_data: pd.DataFrame) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    try:
        X_test = test_data.drop(columns=['label']).values
        y_test = test_data['label'].values
        y_pred = model.predict(X_test)
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logging.info("Model evaluation completed successfully.")
        return metrics_dict
    except Exception as e:
        logging.error("Error during model evaluation: %s", e)
        raise

def save_metrics(metrics: Dict[str, float], report_path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info("Evaluation metrics saved to %s", report_path)
    except Exception as e:
        logging.error("Error saving metrics to %s: %s", report_path, e)
        raise

def main() -> None:
    """Main function to orchestrate model evaluation."""
    try:
        model = load_model("models/random_forest_model.pkl")
        test_data = load_test_data("data/interim/test_bow.csv")
        metrics = evaluate_model(model, test_data)
        save_metrics(metrics, "reports/evalution_metrics.json")
        print("Model evaluation completed and metrics saved successfully.")
        logging.info("Model evaluation pipeline completed successfully.")
    except Exception as e:
        logging.critical("Model evaluation pipeline failed: %s", e)
        raise

if __name__ == "__main__":
    main()