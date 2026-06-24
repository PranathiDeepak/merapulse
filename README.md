# Heart Disease Prediction

An end-to-end machine learning web application that predicts heart disease risk from clinical features. Built with an ensemble of Logistic Regression and Random Forest classifiers, served through a Flask web interface.

## About

This project was built during my Bachelor's degree as a hands-on introduction to applied machine learning and web deployment. The model takes 13 clinical parameters as input and classifies whether a patient is at risk of heart disease.

**Key aspects:**
- Multiple classification models trained and evaluated (Logistic Regression, Random Forest)
- Ensemble model using soft voting for improved prediction accuracy
- End-to-end deployment as a Flask web application with a form-based UI

## Tech Stack

- **Python** — scikit-learn, Pandas, NumPy
- **Flask** — web framework and routing
- **HTML / CSS** — frontend templates

## Project Structure

```
merapulse/
├── app.py          # Flask app — trains model on startup and serves predictions
├── model.py        # Standalone training script (optional)
├── dataset.csv     # Cleveland Heart Disease dataset (303 patients)
├── templates/      # HTML pages (prediction form, result pages)
└── static/         # CSS and static assets
```

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Start the app**
```bash
python app.py
```
The model trains automatically on startup — no separate training step needed.

**3. Open in browser**

Go to `http://localhost:5000`, fill in the patient details, and click **Predict**.

## Input Features

| Feature    | Description                                               |
|------------|-----------------------------------------------------------|
| Age        | Age in years                                              |
| Sex        | 1 = male, 0 = female                                      |
| CP         | Chest pain type (0–3)                                     |
| Trestbps   | Resting blood pressure (mm Hg)                            |
| Chol       | Serum cholesterol (mg/dl)                                 |
| FBS        | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)    |
| Restecg    | Resting ECG results (0–2)                                 |
| Thalach    | Maximum heart rate achieved                               |
| Exang      | Exercise-induced angina (1 = yes, 0 = no)                 |
| Oldpeak    | ST depression (exercise vs rest)                          |
| Slope      | Slope of peak exercise ST segment (0–2)                   |
| CA         | Major vessels coloured by fluoroscopy (0–3)               |
| Thal       | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible)|

## Dataset

[Cleveland Heart Disease Dataset](https://www.openml.org/d/43) — 303 patient records, 13 clinical features, binary classification target.
