# Imports

# pandas
import pandas as pd
from pandas import DataFrame

# numpy, matplotlib, seaborn
#%matplotlib inline
import numpy as np
import csv
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#get titanic and test csv files as a DataFrame
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

#first method prediction
trainX_df = train_df.drop(["Age","SibSp","Parch","Ticket",
	"Fare","Cabin","Embarked","Pclass","Name","PassengerId","Survived"],axis=1)
trainY_df = train_df.drop(["Age","SibSp","Parch","Ticket",
	"Fare","Cabin","Embarked","Pclass","Name","PassengerId","Sex"],axis=1)
testX_df = test_df.drop(["Age","SibSp","Parch","Ticket",
	"Fare","Cabin","Embarked","Pclass","Name","PassengerId"],axis=1)

for index,row in trainX_df.iterrows():
	if row['Sex'] == 'male':
		row['Sex'] = 0
	else:
		row['Sex'] = 1
for index,row in testX_df.iterrows():
	if row['Sex'] == 'male':
		row['Sex'] = 0
	else:
		row['Sex'] = 1
#SVM
sv = 1
if sv == 1:
	clf = SVC()
	clf.fit(trainX_df,trainY_df)
	testY_df = clf.predict(testX_df)
	testY_df = pd.DataFrame(data=testY_df)
	testY_df.to_csv('../data/titanicSVM.csv')

#Logistic Regression
log = 0
if log == 1:
	logreg = LogisticRegression()
	logreg.fit(trainX_df, trainY_df)
	Y_pred = logreg.predict(testX_df)

# Random Forests
rfor = 0
if rfor == 1:
	random_forest = RandomForestClassifier(n_estimators=100)
	random_forest.fit(trainX_df, trainY_df)
	Y_pred = random_forest.predict(testX_df)
	print(random_forest.score(trainX_df,trainY_df))