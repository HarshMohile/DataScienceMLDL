from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.externals import joblib




app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	iris = pd.read_csv("iris.csv")
	neigh = KNC(n_neighbors= 3)
	neigh.fit(iris.iloc[:,0:4],iris.iloc[:,4])


    

	
	

	if request.method == 'POST':
		sl = request.form['sl']
		sw = request.form['sw']
		pl = request.form['pl']
		pw = request.form['pw']
		pred = pd.DataFrame([[sl,sw,pl,pw]])
		pred.columns=['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']
		my_pred = neigh.predict(pred)
	return render_template('result.html',prediction=my_pred)



if __name__ == '__main__':
	app.run(debug=True,use_reloader=False)