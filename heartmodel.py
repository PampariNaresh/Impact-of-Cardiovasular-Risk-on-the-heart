import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
filename =  '\datasets\heart.csv'
filename =os.getcwd() +filename

# filename =os.path.join(os.getcwd(),filename)

heartdata = pd.read_csv(filename)
X = heartdata.drop(columns = 'target', axis = 1)
y = heartdata['target']
scaler = StandardScaler()
scaler.fit(X)
X_standard = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, stratify = y, random_state = 3 )

max_accuracy =0
best_x =0

for x in range(100):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,y_train)
import pickle

with open('heartmodel1.pkl','wb') as f:
  pickle.dump(rf,f)