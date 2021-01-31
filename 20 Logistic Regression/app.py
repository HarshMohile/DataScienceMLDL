from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import pickle
import statsmodels.formula.api as sm
#from sklearn.externals import joblib
import sys

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	claimants=pd.read_csv("claimants.csv")
	claimants.drop(["CASENUM"],axis=1,inplace=True)
	claimants.iloc[:,0:4]=claimants.iloc[:,0:4].apply(lambda x:x.fillna(x.mode()[0]))
	claimants.CLMAGE=claimants.CLMAGE.fillna(claimants.CLMAGE.mean())


	model=sm.logit('ATTORNEY~CLMSEX+CLMINSUR+SEATBELT+CLMAGE+LOSS',data=claimants).fit()
	
	if request.method == 'POST':
		gender = int(request.form['clmsex'])
		ins = int(request.form['clmins'])
		seat = int(request.form['seat'])
		age = int(request.form['age'])
		loss = float(request.form['loss'])
		# predict_dict = {}
		# predict_dict['CLMSEX'] = [int(gender)]
		# predict_dict['CLMINSUR'] = [int(ins)]
		# predict_dict['SEATBELT'] = [int(seat)]
		# predict_dict['CLMAGE'] = [int(age)]
		# predict_dict['LOSS'] = [int(loss)]
		# print()
		pred = [[gender,ins,seat,age,loss]]
		pred = pd.DataFrame(([[gender,ins,seat,age,loss]]))
		print(type(pred),file = sys.stdout)
		pred.columns=['CLMSEX','CLMINSUR','SEATBELT','CLMAGE','LOSS']
		print(pred,file = sys.stdout)
		my_pred=model.predict(pred)
		print(my_pred,file = sys.stdout)
		print(type(my_pred),file = sys.stdout)
		return render_template('result.html',prediction=my_pred[0])



if __name__ == '__main__':
	app.run(debug=True,use_reloader=False)
	
