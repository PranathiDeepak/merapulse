# Heart Disease Prediction using Machine Learning

This is a simple end-to-end machine learning project that predicts the risk of heart disease using basic classification models. The project demonstrates the full workflow from data preprocessing and model training to deployment as a REST API with basic explainability.

## Description
The goal of this project is to build a machine learning model that can classify whether a person is likely to have heart disease based on clinical features. The trained model is exposed through a Flask-based API for real-time predictions, and SHAP is used to provide feature importance for model interpretability.

## Tools and Technologies
- Python  
- scikit-learn  
- Flask  
- SHAP  
- Pandas  
- NumPy  

## Key Features
- Data preprocessing and feature engineering  
- Training of simple classification models  
- Model evaluation using standard metrics  
- REST API for real-time inference  
- Explainable predictions using SHAP  

## How to Run
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
2. Train the model:
   python train.py
3. Start the Flask server:
   python app.py
4. Send input features as a POST request to get predictions.
