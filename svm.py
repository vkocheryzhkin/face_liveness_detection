from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

usecols = [*range(0, 68)]
df = pd.read_csv('combine.csv', header=None)#, usecols=usecols)

X = df.drop(df.columns[68], axis=1).values
y =  df[df.columns[68]].values

clf = svm.SVC()
t = clf.fit(X, y)  

print(clf)

predicted = clf.predict(X)

score = accuracy_score(y, predicted)
print(score)

idx = 16
t = df[idx:idx+1]
print(t)
print(X[idx])
p = clf.predict([X[idx]])
print(p)