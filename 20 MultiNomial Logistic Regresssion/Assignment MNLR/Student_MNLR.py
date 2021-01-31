### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 


stud = pd.read_csv("D:\\360Assignments\\Submission\\20 MultiNomial Logistic Regresssion\\Assignment MNLR\\mdata.csv")
stud.head(10)


stud.describe()
stud.value_counts()

stud.columns

# Dropping the id and Unnamed 
st1 = stud.drop(['Unnamed: 0','id'],axis=1)

st1.rename(columns={"female":"gender"},inplace=True)

# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "gender", y = "math", data = st1)
sns.boxplot(x = "gender", y = "read", data = st1)
sns.boxplot(x = "gender", y = "read", data = st1)


#Countplot
sns.countplot(x='prog',data=st1)
# Barplot
sns.barplot(x = "gender", y = "math", data = st1)

# Stripplot
sns.stripplot(x = "gender", y = "math", data = st1)
sns.stripplot(x = "gender", y = "math", data = st1)

# violinplot
sns.violinplot(x = "honors", y = "math", data = st1)

# Pairplot
sns.pairplot(st1, hue = "honors") 
sns.pairplot(st1, hue = "prog")

# Correlation values between each independent features
st1.corr() 

'''
             read     write      math   science
read     1.000000  0.596776  0.662280  0.630158
write    0.596776  1.000000  0.617449  0.570442
math     0.662280  0.617449  1.000000  0.630733
science  0.630158  0.570442  0.630733  1.000000
'''

# Label encoding by categorical data
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
st1["gender"] = lb.fit_transform(st1["gender"])
st1["ses"] = lb.fit_transform(st1["ses"])
st1["schtyp"] = lb.fit_transform(st1["schtyp"])
st1["honors"] = lb.fit_transform(st1["honors"])



# X and y 
st1.columns

X = st1.drop(columns={"prog"},axis=1)
y = st1["prog"]


# Train test split
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
 
from sklearn.preprocessing import MinMaxScaler

scaler =MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

#Same for X_test .Beacuse to avoid leakage 
X_test = scaler.transform(X_test)

# Normalizing the data  (removed the prog , normalize other and then joined them)
'''
st2 =st1.drop(["prog"],axis=1)

def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

st2_norm = norm_func(st2)

prog = st1['prog']
st2_norm = st2_norm.join(prog)
'''

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg")

model.fit(X_train,y_train)

test_predict = model.predict(X_test) # Test predictions

# Test accuracy   0.65
accuracy_score(y_test, test_predict)

train_predict = model.predict(X_train) # Train predictions 

# Train accuracy 0.61
accuracy_score(y_train, train_predict) 

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test, test_predict))

print(classification_report(y_test, test_predict))

'''
              precision    recall  f1-score   support

    academic       0.71      0.87      0.78       105
     general       0.47      0.20      0.28        45
    vocation       0.58      0.60      0.59        50

    accuracy                           0.65       200
   macro avg       0.59      0.56      0.55       200
weighted avg       0.62      0.65      0.62       200
'''

# Imprving the model by gridSearch

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100] }

grid = GridSearchCV(LogisticRegression(multi_class = "multinomial"),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)



grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test,grid_predictions))

'''
[[ 36  11]
 [  6 103]]
'''
print(classification_report(y_test,grid_predictions))
