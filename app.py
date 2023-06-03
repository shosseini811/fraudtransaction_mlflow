import numpy as np
import pandas as pd
import matplotlib #
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import KFold
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

# Set the experiment name to an experiment with a descriptive name
mlflow.set_experiment('fraud_detection_experiment')

# Start a new run in this experiment
with mlflow.start_run():
    # Define the model 
    clf = LogisticRegression(random_state=42)

    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = clf.predict(X_test)
    
    # Log model
    mlflow.sklearn.log_model(clf, "model")
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, predictions))
    
    # Print out metrics
    print("Model accuracy: ", accuracy_score(y_test, predictions))
    print("Confusion Matrix: \n", confusion_matrix(y_test, predictions))
    print("Classification Report: \n", classification_report(y_test, predictions))
