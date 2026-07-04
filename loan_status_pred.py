import pandas as pd
import numpy as np

data=pd.read_csv(r"C:\Users\srish\Downloads\ml\loan_prediction.csv")
print(data)

print(data.isnull().sum()*100 / len(data))

data = data.drop('Loan_ID',axis=1)

columns = ['Gender','Dependents','LoanAmount','Loan_Amount_Term']
data = data.dropna(subset=columns)
print(data.isnull().sum()*100 / len(data))


print(data['Self_Employed'].mode()[0])

data['Self_Employed']=data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
print(data)

print(data.isnull().sum()*100 / len(data))





data['Credit_History']=data['Credit_History'].fillna(data['Credit_History'].mode()[0])
print(data['Credit_History'].unique())





print(data['Dependents'].unique())


data['Dependents']=data['Dependents'].replace(to_replace="3+",value="4")
print(data['Dependents'].unique())


data['Married']=data['Married'].map({'Yes':'1','No':'0'}).astype('int')
print(data)

data['Self_Employed']=data['Self_Employed'].map({'Yes':'1','No':'0'}).astype('int')


data['Loan_Status']=data['Loan_Status'].map({'Y':'1','N':'0'}).astype('int')
data['Gender']=data['Gender'].map({'Male':'1','Female':'0'}).astype('int')
data['Education']=data['Education'].map({'Graduate':'1','Not Graduate':'0'}).astype('int')
data['Property_Area']=data['Property_Area'].map({'Semiurban':'1','Rural':'0','Urban':'2'}).astype('int')




print(data['Property_Area'].unique())
print(data['Education'].unique())

print(data)



X=data.drop('Loan_Status',axis=1)
y=data['Loan_Status']


cols=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']

from sklearn.preprocessing import StandardScaler
st=StandardScaler()

X[cols]=st.fit_transform(X[cols])
print(X)


# no do k folf cross validation


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


model_df={}
def model_val(model,X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,
                                                   test_size=0.20,
                                                   random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f"{model} accuracy is {accuracy_score(y_test,y_pred)}")
    
    score = cross_val_score(model,X,y,cv=5)
    print(f"{model} Avg cross val score is {np.mean(score)}")
    model_df[model]=round(np.mean(score)*100,2)


# logistic regression

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model_val(model,X,y)



# support vecotr machine
from sklearn import svm
model = svm.SVC()
model_val(model,X,y)


# decison tree classifier

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model_val(model,X,y)


# RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier
model =RandomForestClassifier()
model_val(model,X,y)


# gradoent boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
model =GradientBoostingClassifier()
model_val(model,X,y)

print(model_df)



# hyperparameytr tuning

from sklearn.model_selection import RandomizedSearchCV

log_reg_grid={'C':np.logspace(-4,4,20),'solver':['liblinear']}

# loh=gistic

rs_log_reg=RandomizedSearchCV(LogisticRegression(),param_distributions=log_reg_grid,n_iter=20,cv=5,verbose=True)

rs_log_reg.fit(X,y)

print(rs_log_reg.best_score_)
print(rs_log_reg.best_params_)


# svm

svc_grid = {'C':[0.25,0.50,0.75,1],"kernel":["linear"]}

rs_svc=RandomizedSearchCV(svm.SVC(),
                  param_distributions=svc_grid,
                   cv=5,
                   n_iter=20,
                  verbose=True)
rs_svc.fit(X,y)


print(rs_svc.best_score_)      # best score is the average score of each hyperparametr combination







# random forest classifier

RandomForestClassifier()

rf_grid={'n_estimators':np.arange(10,1000,10),
  'max_features':['auto','sqrt'],
 'max_depth':[None,3,5,10,20,30],
 'min_samples_split':[2,5,20,50,100],
 'min_samples_leaf':[1,2,5,10]
 }

rs_rf=RandomizedSearchCV(RandomForestClassifier(),
                  param_distributions=rf_grid,
                   cv=5,
                   n_iter=20,
                  verbose=True)

rs_rf.fit(X,y)

print(rs_rf.best_score_)



# save the model

X = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']


rf = RandomForestClassifier(n_estimators=270,
 min_samples_split=5,
 min_samples_leaf=5,
 max_features='sqrt',
 max_depth=5)


rf.fit(X,y)

import joblib

joblib.dump(rf,'loan_status_predict')


model = joblib.load('loan_status_predict')

import pandas as pd
df = pd.DataFrame({
    'Gender':1,
    'Married':1,
    'Dependents':2,
    'Education':0,
    'Self_Employed':0,
    'ApplicantIncome':2889,
    'CoapplicantIncome':0.0,
    'LoanAmount':45,
    'Loan_Amount_Term':180,
    'Credit_History':0,
    'Property_Area':1
},index=[0])

result = model.predict(df)


if result==1:
    print("Loan Approved")
else:
    print("Loan Not Approved")




# gui


from tkinter import *
import joblib
import pandas as pd


def show_entry():
    
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    p7 = float(e7.get())
    p8 = float(e8.get())
    p9 = float(e9.get())
    p10 = float(e10.get())
    p11 = float(e11.get())
    
    model = joblib.load('loan_status_predict')
    df = pd.DataFrame({
    'Gender':p1,
    'Married':p2,
    'Dependents':p3,
    'Education':p4,
    'Self_Employed':p5,
    'ApplicantIncome':p6,
    'CoapplicantIncome':p7,
    'LoanAmount':p8,
    'Loan_Amount_Term':p9,
    'Credit_History':p10,
    'Property_Area':p11
},index=[0])
    result = model.predict(df)
    
    if result == 1:
        Label(master, text="Loan approved").grid(row=31)
    else:
        Label(master, text="Loan Not Approved").grid(row=31)


master =Tk()
master.title("Loan Status Prediction Using Machine Learning")
label = Label(master,text = "Loan Status Prediction",bg = "black",
               fg = "white").grid(row=0,columnspan=2)

Label(master,text = "Gender [1:Male ,0:Female]").grid(row=1)
Label(master,text = "Married [1:Yes,0:No]").grid(row=2)
Label(master,text = "Dependents [1,2,3,4]").grid(row=3)
Label(master,text = "Education").grid(row=4)
Label(master,text = "Self_Employed").grid(row=5)
Label(master,text = "ApplicantIncome").grid(row=6)
Label(master,text = "CoapplicantIncome").grid(row=7)
Label(master,text = "LoanAmount").grid(row=8)
Label(master,text = "Loan_Amount_Term").grid(row=9)
Label(master,text = "Credit_History").grid(row=10)
Label(master,text = "Property_Area").grid(row=11)


e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)

e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)
e7.grid(row=7,column=1)
e8.grid(row=8,column=1)
e9.grid(row=9,column=1)
e10.grid(row=10,column=1)
e11.grid(row=11,column=1)

Button(master,text="Predict",command=show_entry).grid()

mainloop()