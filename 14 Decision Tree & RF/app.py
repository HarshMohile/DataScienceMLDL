from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib




app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("Fraud_check.csv")
	model = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
	model.fit(df.iloc[:,0:5],df.iloc[:,5])

	if request.method == 'POST':
		under = request.form['under']
		ms = request.form['ms']
		city = request.form['city']
		work = request.form['work']
		urban = request.form['urban']

		pred = pd.DataFrame([[under,ms,city,work,urban]])
		pred.columns=['Undergrad','Marital.Status','City.Population','Work.Experience','Urban']
		my_pred = model.predict(pred)
	return render_template('result.html',prediction=my_pred)



if __name__ == '__main__':
	app.run(debug=True)