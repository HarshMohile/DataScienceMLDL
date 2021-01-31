from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib




app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("letterdata.csv")
	
	model=SVC(kernel = "linear")
	model.fit(df.iloc[:,1:17],df.iloc[:,0])

	if request.method == 'POST':
		xbox = request.form['xbox']
		ybox = request.form['ybox']
		width = request.form['width']
		height = request.form['height']
		onpix = request.form['onpix']
		xbar = request.form['xbar']
		ybar = request.form['ybar']
		x2bar = request.form['x2bar']
		y2bar = request.form['y2bar']
		xybar = request.form['xybar']
		x2ybar= request.form['x2ybar']
		xy2bar= request.form['xy2bar']
		xedge= request.form['xedge']
		xedgey= request.form['xedgey']
		yedge= request.form['yedge']
		yedgex= request.form['yedgex']


		pred = pd.DataFrame([[xbox,ybox,width,height,onpix,xbar,ybar,x2bar,y2bar,xybar,x2ybar,xy2bar,xedge,xedgey,yedge,yedgex]])
		pred.columns=['xbox','ybox','width','height','onpix','xbar','ybar','x2bar','y2bar','xybar','x2ybar','xy2bar','xedge','xedgey','yedge','yedgex']
		my_pred = model.predict(pred)
	return render_template('result.html',prediction=my_pred)



if __name__ == '__main__':
	app.run(debug=True,use_reloader=False)
