# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:26:56 2023

@author: nikhilve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading the CSV data
data = pd.read_csv('50_Startups.csv')

#finding unique values.
#print(data['State'].unique())

#finding count of each value.
#print(data['State'].value_counts())#selecting features and result.   OR Vector of DV(Dependent Variables) y, and Matrix of IV(Independent Variables) x



x=data.iloc[:,:-1].values

y=data.iloc[:,-1:].values
#y=y.drop(['Purchased_No'],axis=1)


#Label Encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label=LabelEncoder()
x[:,3]=label.fit_transform(x[:,3])


# doing type of one hot encoding to prevent erroe due to label encoding in country field
#from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
#Encode Country Column
label = LabelEncoder()
x[:,3] = label.fit_transform(x[:,3])
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
x = ct.fit_transform(x)


x=x[:,1:]


#Using train-test split to break the data into training and testing data. test_size= 20%data is reserved for testing  
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#Feature Scaling- Here we are scaling all the data into the same scale.
#can be done before train test split, then it would be easy.
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
#y_test = sc_x.transform(y_test) 


#Importing Linear Regression, regressor will be our model.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Training the regressor.
regressor.fit(x_train, y_train)


#predicting Output:
y_predict = regressor.predict(x_test)

y_predict= sc_y.inverse_transform(y_predict)


#Here c is the W0 value
b=regressor.intercept_
print("W0 value :",b)

#Here b is the w1,w2,w3,w4,w5 values
c=regressor.coef_
print("Respective Values of W1,W2,W3,W4 and W5",c)

d=sc_y.inverse_transform(regressor.predict([[-0.57735,1.36277,-1.11353,-2.21896,-0.136691
]]))
print("Predicted Value :",d)




# Visual representation of the training-set predictions.
plt.scatter(x_train[:, 3], y_train, color='red', label='Actual')
plt.scatter(x_train[:, 3], regressor.predict(x_train), color='blue', label='Predicted')
plt.xlabel("Experience in Years")
plt.ylabel("Salary")
plt.title("Training-Set Graph: Salary vs Exp")
plt.legend()
plt.show()




# Visual representation of test-set predictions.
#The algorithm is adjusting the salaries of under-paid and over-paid employees and giving a visual representation.
plt.scatter(x_test[:, 3], y_test, color='red', label='Actual')
plt.scatter(x_test[:, 3], y_predict, color='blue', label='Predicted')
plt.xlabel("Experience in Years")
plt.ylabel("Salary")
plt.title("Test-Set Graph: Salary vs Exp")
plt.legend()
plt.show()









