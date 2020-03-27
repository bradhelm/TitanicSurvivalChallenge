import numpy as np
import pandas as pd
import os

from sklearn.model_selection import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.neighbors import *
from sklearn.discriminant_analysis import *
from sklearn.linear_model import *

raw_train = pd.read_csv("data/train.csv")

X = raw_train.drop(['Survived','PassengerId','Cabin','Ticket','Fare','Name'], axis=1)
y = raw_train['Survived']

def prepare(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Age'] = df['Age']/17
    df['Age'] = df['Age'].astype(int)

    df['Child'] = (df['Age']==0)
    df['Child'] = df['Child'].astype(int)

    df['Alone'] = (df['SibSp']+df['Parch'])==0
    df['Alone'] = df['Alone'].astype(int)

    for gender in ['male','female']:
        df[gender] = (df['Sex']==gender)
        df[gender] = df[gender].astype(int)

    for embarked in ['C', 'Q', 'S']:
        df[embarked] = (df['Embarked']==embarked)
        df[embarked] = df[embarked].astype(int)

    df = df.drop(['Sex','Embarked','male','C','Q','S'], axis=1)

    df.reset_index(drop=True)
    return df

X = prepare(X)

folds = 8
classifiers = [
    SVC(kernel='poly', degree=3),
    SVC(kernel="rbf", C=0.025, probability=True),
    KNeighborsClassifier(5),
    NuSVC(probability=True),
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    LinearDiscriminantAnalysis(),
    LogisticRegression(max_iter=10000)]

best_clf=None
best_score=-1
for clf in classifiers:
    scores = []
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X, y):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    
    print("Accuracy over ", folds, " folds: ", np.mean(scores), type(clf))
    if best_score<np.mean(scores):
        best_score=np.mean(scores)
        best_clf=clf

print("Best classifier evaluated is ", type(best_clf), " with a mean score of ", best_score)
best_clf.fit(X,y)

raw_test = pd.read_csv("data/test.csv")

X = raw_test.drop(['PassengerId','Cabin','Ticket','Fare','Name'], axis=1)
X = prepare(X)

raw_test['Survived']=best_clf.predict(X)
raw_test[['PassengerId','Survived']].to_csv('data/submission.csv', index=False)

