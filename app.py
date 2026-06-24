from flask import Flask, render_template, request, redirect
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

app = Flask(__name__, static_folder='static')

# Train the ensemble model on startup
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

logreg = LogisticRegression()
rf = RandomForestClassifier(n_estimators=100, random_state=1)

model = VotingClassifier(
    estimators=[('lr', logreg), ('rf', rf)],
    voting='soft'
)
model.fit(X_train, Y_train)

@app.route('/home')
def home():
    return redirect('home.html')

@app.route("/")
def index():
    return render_template('prediction.html')

@app.route('/result', methods=['POST', 'GET'])
def result():
    age      = int(request.form['age'])
    sex      = int(request.form['sex'])
    trestbps = float(request.form['trestbps'])
    chol     = float(request.form['chol'])
    restecg  = float(request.form['restecg'])
    thalach  = float(request.form['thalach'])
    exang    = int(request.form['exang'])
    cp       = int(request.form['cp'])
    fbs      = float(request.form['fbs'])
    oldpeak  = float(request.form['oldpeak'])
    slope    = int(request.form['slope'])
    ca       = int(request.form['ca'])
    thal     = int(request.form['thal'])

    x = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

    s = model.predict(x)

    if s == 0:
        return render_template('nodisease.html')
    else:
        return render_template('file.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

if __name__ == "__main__":
    app.run(debug=True)
