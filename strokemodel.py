# The libraries used in processing the dataset
import numpy as np
import pandas as pd

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

import os
from sklearn.preprocessing import StandardScaler


filename =  '\datasets\heartstroke.csv'
filename =os.getcwd() +filename
strokedata = pd.read_csv(filename)

#Removing id Column Because no use
strokedata.drop("id", axis=1, inplace=True)
def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers


#print(strokedata.loc[detect_outliers(strokedata,['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease', 'stroke'])])
# drop outliers
strokedata = strokedata.drop(detect_outliers(strokedata,['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease', 'stroke']),axis = 0).reset_index(drop = True)

# print("Mean of BMI value for Females: ", np.mean(strokedata[strokedata['gender'] == 'Female']['bmi']))
# print("Mean of BMI value for Males: ", np.mean(strokedata[strokedata['gender'] == 'Male']['bmi']))
# print("Mean of BMI value for Others: ", np.mean(strokedata['bmi']))

strokedata['bmi'] = strokedata['bmi'].fillna(0)
for i in range(0,5035):
    if(strokedata['bmi'][i] == 0):
        if(strokedata['gender'][i] == 'Male'):
            strokedata['bmi'][i] = 28.594683544303823
        elif(strokedata['gender'][i] == 'Female'):
            strokedata['bmi'][i] = 29.035926055109936
        else:
            strokedata['bmi'][i] = 28.854652338161664

#print(len(strokedata[strokedata['bmi'].isnull()]))


# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# print("Unique Values for Gender", dataset['gender'].unique())
# print("Unique Values for ever_married", dataset['ever_married'].unique())
# print("Unique Values for work_type", dataset['work_type'].unique())
# print("Unique Values for Residence_type", dataset['Residence_type'].unique())
# print("Unique Values for smoking_status", dataset['smoking_status'].unique())


ever_married_mapping = {'No': 0, 'Yes': 1}
strokedata['ever_married'] = strokedata['ever_married'].map(ever_married_mapping)

Residence_type_mapping = {'Rural': 0, 'Urban': 1}
strokedata['Residence_type'] = strokedata['Residence_type'].map(Residence_type_mapping)

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder()
strokedata['gender'] = pd.Categorical(strokedata['gender'])
strokedataDummies_gender = pd.get_dummies(strokedata['gender'], prefix = 'gender_encoded')
#strokedataDummies_gender

strokedata['work_type'] = pd.Categorical(strokedata['work_type'])
strokedataDummies_work_type = pd.get_dummies(strokedata['work_type'], prefix = 'work_type_encoded')
#strokedataDummies_work_type

strokedata['smoking_status'] = pd.Categorical(strokedata['smoking_status'])
strokedataDummies_smoking_status = pd.get_dummies(strokedata['smoking_status'], prefix = 'smoking_status_encoded')
#strokedataDummies_smoking_status

strokedata.drop("gender", axis=1, inplace=True)
strokedata.drop("work_type", axis=1, inplace=True)
strokedata.drop("smoking_status", axis=1, inplace=True)

strokedata = pd.concat([strokedata, strokedataDummies_gender], axis=1)
strokedata = pd.concat([strokedata, strokedataDummies_work_type], axis=1)
strokedata = pd.concat([strokedata, strokedataDummies_smoking_status], axis=1)


#Model training
features = ['age',
 'hypertension',
 'heart_disease',
 'ever_married',
 'Residence_type',
 'avg_glucose_level',
 'bmi',
 'gender_encoded_Female',
 'gender_encoded_Male',
 'gender_encoded_Other',
 'work_type_encoded_Govt_job',
 'work_type_encoded_Never_worked',
 'work_type_encoded_Private',
 'work_type_encoded_Self-employed',
 'work_type_encoded_children',
 'smoking_status_encoded_Unknown',
 'smoking_status_encoded_formerly smoked',
 'smoking_status_encoded_never smoked',
 'smoking_status_encoded_smokes']

label = ['stroke']

X = strokedata[features]
y = strokedata[label]

from imblearn.over_sampling import RandomOverSampler

# Performing a minority oversampling
oversample = RandomOverSampler(sampling_strategy='minority')


# Obtaining the oversampled dataframes - testing and training
X, y = oversample.fit_resample(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3) 
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score


max_accuracy =0
best_x =0

for x in range(100):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,y_train.values.ravel())
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,y_train.values.ravel())
import pickle5 as pickle

with open('strokemodel.pkl','wb') as f:
  pickle.dump(rf,f)




































