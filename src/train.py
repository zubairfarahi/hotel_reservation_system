import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Load the data from CSV files
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
valid_df = pd.read_csv("data/valid.csv")

# Define features and target variable for training, test, and validation
X_train = train_df.drop(columns=['booking_status'])
y_train = train_df['booking_status']

X_test = test_df.drop(columns=['booking_status'])
y_test = test_df['booking_status']

X_valid = valid_df.drop(columns=['booking_status'])
y_valid = valid_df['booking_status']

# Define multiple classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Classifier": SVC(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "LGBM": LGBMClassifier(random_state=42)
}

# Start an MLflow experiment
mlflow.set_experiment("hotel_reservation_experiment")

# Train and evaluate each model
results = {}
for name, model in classifiers.items():
    with mlflow.start_run():
        print(f"Training {name}...")
        
        # Log the model name and parameters in MLflow
        mlflow.log_param("model", name)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Calculate accuracy and log it in MLflow
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the classification report and confusion matrix as artifacts
        classification_report_str = classification_report(y_test, y_pred)
        confusion_matrix_str = str(confusion_matrix(y_test, y_pred))
        
        with open(f"models/{name.replace(' ', '_').lower()}_classification_report.txt", 'w') as f:
            f.write(classification_report_str)
        with open(f"models/{name.replace(' ', '_').lower()}_confusion_matrix.txt", 'w') as f:
            f.write(confusion_matrix_str)
        
        mlflow.log_artifact(f"models/{name.replace(' ', '_').lower()}_classification_report.txt")
        mlflow.log_artifact(f"models/{name.replace(' ', '_').lower()}_confusion_matrix.txt")
        
        # Log the trained model to MLflow
        input_example = X_train.iloc[0].to_dict()  # Use the first row from training data as the input example
        if name == "XGBoost":
            mlflow.xgboost.log_model(model, name, input_example=input_example)
        elif name == "LGBM":
            mlflow.lightgbm.log_model(model, name, input_example=input_example)
        else:
            mlflow.sklearn.log_model(model, name, input_example=input_example)
        
        # Save the model locally as well
        model_path = f"models/{name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_path)
        
        results[name] = accuracy

# Print the best model
best_model = max(results, key=results.get)
print(f"Best model: {best_model} with accuracy {results[best_model]:.4f}")

# Log the best model result to MLflow
with mlflow.start_run():
    mlflow.log_param("best_model", best_model)
    mlflow.log_metric("best_model_accuracy", results[best_model])
