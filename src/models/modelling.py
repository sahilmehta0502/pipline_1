import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
import pickle

train_data = pd.read_csv("data/interim/train.csv")
X_train = train_data.drop(columns=['label']).values
y_train = train_data['label'].values

model= RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pickle.dump(model, open("models/random_forest_model.pkl", "wb"))
# Save the trained model to a file
print("Model trained and saved successfully.")


