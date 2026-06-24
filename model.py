# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# Load dataset
dataset = pd.read_csv('dataset.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

# Base models
logreg = LogisticRegression()
rf = RandomForestClassifier(n_estimators=100, random_state=1)

# Ensemble: soft voting combines probability outputs of both models
ensemble = VotingClassifier(
    estimators=[('lr', logreg), ('rf', rf)],
    voting='soft'
)
ensemble.fit(X_train, Y_train)

# Evaluation
y_pred = ensemble.predict(X_test)
y_prob = ensemble.predict_proba(X_test)[:, 1]
print("Accuracy:", round(accuracy_score(Y_test, y_pred), 4))
print("AUC:     ", round(roc_auc_score(Y_test, y_prob), 4))
