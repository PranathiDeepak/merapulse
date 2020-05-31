from flask import Flask, render_template, url_for, request,redirect
import os 
import numpy as np 
import pickle 
app = Flask(__name__, static_folder='static')
@app.route('/home')
def home():
	return redirect('home.html')
 
@app.route("/") 
def index():     
  return render_template('prediction.html')  
 
@app.route('/result', methods=['POST', 'GET']) 
def result():     
  age = int(request.form['age'])     
  sex = int(request.form['sex'])     
  trestbps = float(request.form['trestbps'])     
  chol = float(request.form['chol'])     
  restecg = float(request.form['restecg'])     
  thalach = float(request.form['thalach'])     
  exang = int(request.form['exang'])     
  cp = int(request.form['cp'])     
  fbs = float(request.form['fbs']) 
  oldpeak = float(request.form['oldpeak'])  
  slope = int(request.form['slope'])  
  ca = int(request.form['ca'])  
  thal = int(request.form['thal'])      
  x = np.array([age, sex, cp, trestbps, chol, fbs, restecg,                   
           thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
  print(x)

  
  model = pickle.load(open('model.pkl','rb'))
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
     
     
