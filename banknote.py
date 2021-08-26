# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 13:39:36 2021

@author: sachin h s
"""

import pandas as pd
import numpy as np
df = pd.read_csv('BankNote_Authentication.data')
X=df.iloc[:,:-1]
y=df.iloc[:,-1]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
#from sklearn.metrics import accuracy_score
#score=accuracy_score(y_test,y_pred)



import pickle
pickle.dump(classifier, open('classifier.pkl', 'wb'))
##pickle_out = open("classifier.pkl","wb")
#pickle.dump(classifier, pickle_out)
#pickle_out.close()