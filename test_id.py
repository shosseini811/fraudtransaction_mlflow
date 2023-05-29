import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import mlflow
import mlflow.sklearn

data_path = "creditcard.csv"
df = pd.read_csv('creditcard.csv')


df = df.drop("Time", axis=1)

# Separate the features from the target
X = df.iloc[:, :-1]  # all features
y = df['Class']  # target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# replace with your run_id
run_id = "221f510410c94fda9b1631860a95450f"

model_uri = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))