#   Titanic: Machine Learning from Disaster
#   By Barry Guglielmo


#   This Code was written to submit to Kaggle for their intro compitition
#   It takes the Titanic Data and creates a support Vector Machine for
#   a binary classification problem to guesses who will survive the Titanic accident.

#   This was my first attempt so I kept it simple. There is much more that I could
#   Do to improve the accuracy of my model such as including their names, and pick up location.


#Dependencies
import pandas as pd
from sklearn import svm
import numpy as np
#Path and training/test Data
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
#Replace NaN age with mean age
mean_age = train['Age'].mean()
train['Age'].fillna(mean_age, inplace=True)
#Replace male and female with 1 or 0
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
#The indacies of the columns for use (Features)
X = train.iloc[:,[2,4,5,6,7,9]]
#Survived or Not
y = train.iloc[:,[1]]
y = np.ravel(y)
#Make the model
model = svm.SVC()
#fit the Model
model.fit(X,y)
#print score
print(model.score(X,y))

#unknowns to test
mean_age2 = test['Age'].mean()
mean_fare = test['Fare'].mean()
test['Age'].fillna(mean_age2, inplace=True)
test['Fare'].fillna(mean_fare, inplace=True)
test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})
unknown = test.iloc[:,[1,3,4,5,6,8]]
ID = test['PassengerId']
predict_unknown = model.predict(unknown)


#Write out my prediciton to csv
pred = pd.DataFrame(data = predict_unknown)
pred.to_csv('./Prediction.csv')
