# Imports

# pandas
import pandas as pd
from pandas import DataFrame

# numpy, matplotlib
import numpy as np
import csv
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifiers

#get titanic and test csv files as a DataFrame
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")
data = pd.concat([train_df,test_df],ignore_index=True)
# If age is less than 1, we return 1. Else, we return the original age.
def normalize_age_below_one(age):
    if age < 1:
        return 1
    else:
        return age
# Those with age <1, changed to 1

def sex(value):
    if value == "male":
        return 0
    else:
        return 1

def embarked(value):
    if value == "C":
        return 0
    elif value =="Q":
        return 1
    else:
        return 2

data['Age'] = data['Age'].apply(normalize_age_below_one)
data['Age'] = data['Age'].fillna(28)
data['Fare'] = data['Fare'].fillna(14.45)
data['Sex'] = data['Sex'].apply(sex)
data['Embarked'] = data['Embarked'].apply(embarked)
data["TicketClean"] = data["Ticket"].str.extract('(\d{2,})', expand=True)
data["TicketClean"] = data["Ticket"].str.extract('(\d{3,})', expand=True)
data["TicketClean"] = data["TicketClean"].apply(pd.to_numeric)
med1=data["TicketClean"].median()
med2=data["TicketClean"].median()+data["TicketClean"].std()
med3=data["TicketClean"].median()-data["TicketClean"].std()
data.set_value(179, 'TicketClean', int(med1))
data.set_value(271, 'TicketClean', int(med1))
data.set_value(302, 'TicketClean', int(med1))
data.set_value(597, 'TicketClean', int(med1))
data.set_value(772, 'TicketClean', int(med2))
data.set_value(841, 'TicketClean', int(med2))
data.set_value(1077, 'TicketClean', int(med2))
data.set_value(1193, 'TicketClean', int(med2))

#first method prediction
trainX_df = data.drop(["Embarked","SibSp","Parch","Ticket"
	,"Cabin","Name","Survived","PassengerId"],axis=1)
trainX_df = trainX_df[:][:891]
trainY_df = data.drop(["Age","SibSp","Parch","Ticket",
	"Fare","Cabin","Embarked","Pclass","Name","PassengerId","Sex","TicketClean"],axis=1)
trainY_df = trainY_df[:][:891]
testX_df = data.drop(["Embarked","SibSp","Parch","Ticket",
	"Cabin","Name","PassengerId","Survived"],axis=1)
testX_df = testX_df[:][891:]

#SVM
sv = 0
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
	testY_df = pd.DataFrame(data=Y_pred)
	testY_df.to_csv('../data/titanicSVM.csv')
	print(random_forest.score(trainX_df,trainY_df))