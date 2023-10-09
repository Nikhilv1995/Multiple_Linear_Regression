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
#data['State'].value_counts()


#selecting features and result.   OR Vector of DV(Dependent Variables) y, and Matrix of IV(Independent Variables) x
X=data.iloc[:,:-1].values

y=data.iloc[:,-1:].values
#y=y.drop(['Purchased_No'],axis=1)

#Label Encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label=LabelEncoder()
X[:,3]=label.fit_transform(X[:,3])

#performing onehot encoding on label encoded column. Also dropping the first column of onehot-encoded values to avoid dummy variable trap.
onehotencoder= OneHotEncoder(drop='first', sparse_output=False)
X_encoded= onehotencoder.fit_transform(X[:,3].reshape(-1,1))#reshaping into 2d array using reshape.


# Concatenate the original X with the one-hot encoded features
X = np.concatenate((X[:,0:3 ], X_encoded), axis=1)


#Using train-test split to break the data into training and testing data. test_size= 20%data is reserved for testing  
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#Feature Scaling- Here we are scaling all the data into the same scale.
#can be done before train test split, then it would be easy.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
#y_test = sc_x.transform(y_test) 


#Importing Linear Regression, regressor will be our model.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Training the regressor.
regressor.fit(X_train, y_train)



#predicting Output:
y_predict = regressor.predict(X_test)

y_predict= sc_y.inverse_transform(y_predict)


#Here b is the W0 value
b=regressor.intercept_
print("W0 value :",b)

#Here c is the w1,w2,w3,w4,w5 values
c=regressor.coef_
print("Respective Values of W1,W2,W3,W4 and W5",c)

d=sc_y.inverse_transform(regressor.predict([[-0.227907	,1.13222	,-0.922749,	-0.57735,	1.36277

]]))
print("Predicted Value :",d)

#Checking accuracy
from sklearn.metrics import r2_score

# Calculate R-squared
##R squared ranges from 0 to 1, 1 is considered as the perfect fit.
r_squared = r2_score(y_test, y_predict)

print("R-squared:", r_squared)




# Plotting Training Predictions vs. Training Data
#plt.figure(figsize=(10, 6))
plt.scatter(y_train, regressor.predict(X_train), color='blue')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], linestyle='--', color='red', linewidth=2)
plt.xlabel('Actual Values (Training Data)')
plt.ylabel('Predicted Values (Training Predictions)')
plt.title('Training Predictions vs. Training Data')
plt.show()

# Plotting Test Predictions vs. Test Data
#plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_predict, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='purple', linewidth=2)
plt.xlabel('Actual Values (Test Data)')
plt.ylabel('Predicted Values (Test Predictions)')
plt.title('Test Predictions vs. Test Data')
plt.show()


# Visual representation of the training-set predictions.
plt.scatter(X_train[:, [1]], y_train, color='red', label='Actual')
plt.scatter(X_train[:, [1]], regressor.predict(X_train), color='blue', label='Predicted')
plt.xlabel("Experience in Years")
plt.ylabel("Salary")
plt.title("Training-Set Graph: Salary vs Exp")
plt.legend()
plt.show()




# Visual representation of test-set predictions.
#The algorithm is adjusting the salaries of under-paid and over-paid employees and giving a visual representation.
plt.scatter(X_test[:,[1]], y_test, color='red', label='Actual')
plt.scatter(X_test[:, [1]], y_predict, color='blue', label='Predicted')
plt.xlabel("Experience in Years")
plt.ylabel("Salary")
plt.title("Test-Set Graph: Salary vs Exp")
plt.legend()
plt.show()









