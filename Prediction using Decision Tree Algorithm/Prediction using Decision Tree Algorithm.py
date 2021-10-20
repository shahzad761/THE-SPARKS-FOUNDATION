# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:11:41 2021

@author: Dell
"""

import pandas as pd
import seaborn as sns
sns.set(style='whitegrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import classification_report

from sklearn import metrics



df = pd.read_csv('Iris.csv',index_col='Id')
x = df.drop('Species',axis=1)
y = df.Species

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3, random_state=12)
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(xtrain,ytrain)
plot_tree(dt,feature_names=xtrain.columns,class_names=ytrain.unique())
plt.show()
pred = dt.predict(xtest)


print(pred[0:5])
print(ytest[0:5])
print( "Accuracy: ", str(metrics.accuracy_score(ytest, pred)*100)+"%")
print(classification_report(ytest,pred))

