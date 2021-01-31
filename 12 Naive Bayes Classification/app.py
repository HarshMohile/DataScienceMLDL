from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib




app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("Diabetes.csv")
	
	Dgnb = GaussianNB()
    
	
	model_gnb = Dgnb.fit(df.iloc[:,0:8],df.iloc[:,8])
    
	if request.method == 'POST':
		preg = request.form['preg']
		Glucose = request.form['glucose']
		Bp = request.form['bp']
		Skin = request.form['skin']
		Insulin = request.form['insulin']
		BMI = request.form['bmi']
		DiabetesPF = request.form['dpf']
		Age = request.form['age']
		pred = pd.DataFrame([[preg,Glucose,Bp,Skin,Insulin,BMI,DiabetesPF,Age]])
		pred.columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
		my_pred = model_gnb.predict(pred)
	return render_template('result.html',prediction=my_pred)



if __name__ == '__main__':
	app.run(debug=True,use_reloader=False)
