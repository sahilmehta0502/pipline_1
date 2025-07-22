from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import pandas as pd
import pickle   
import json

# Load the trained model from file
model = pickle.load(open("models/random_forest_model.pkl", "rb"))
# Load test data from interim features file
test_data = pd.read_csv("data/interim/test_bow.csv")
# Separate features and labels
X_test = test_data.drop(columns=['label']).values
y_test = test_data['label'].values  

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
metrics_dict = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_pred)
}

# Save metrics to a JSON report
with open("reports/evalution_metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)
print("Model evaluation completed and metrics saved successfully.")