from flask import Flask, render_template, request

import numpy as np
import os
import warnings
import pandas as pd
app = Flask(__name__)



from joblib import dump, load
strokefilename = 'strokemodel1.joblib'
strokefilename =os.path.join(os.path.dirname(os.path.abspath(strokefilename)),strokefilename)
global strokemodel
strokemodel = load(open(strokefilename, 'rb'))


heartfilename = 'heartmodel1.joblib'
heartfilename  =os.path.join(os.path.dirname(os.path.abspath(heartfilename)),heartfilename )
global heartmodel

heartmodel = load(open(heartfilename, 'rb'))


@app.route("/")
def home():
  return render_template('home.html')


@app.route('/heartdisease', methods=['GET','POST'])
def heartdisease():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = request.form.get('gender')
        cp = request.form.get('chestpaintype')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['serumcholesterol'])
        fbs = request.form.get('fastbloodsuger')
        restecg = int(request.form['restingegc'])
        thalach = int(request.form['maxheartrate'])
        exang = request.form.get('exerciseinducedanginal')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        my_prediction = heartmodel.predict(data)
       
        return render_template('result.html', prediction=my_prediction)
    elif request.method == 'GET':
        return render_template('heartdiease.html')
    else:
        return render_template('home.html') 
    
@app.route('/strokeprediction', methods=['GET','POST'])
def strokeprediction():   
    if request.method == 'POST':
        age = int(request.form['age'])
        print(age)
        gender = float(request.form.get('gender'))
        print(gender)
        gender_Male=0
        gender_Other=0
        if(gender==1):
            gender_Male=1
            print(gender_Male)
        elif(gender==2):
             gender_Other=1
             print(gender_Other)
        elif(gender==0):
            gender_Male=0
            print(gender_Male)

        hypertension_1 = int(request.form.get('hypertension'))
        
        heart_disease_1 = int(request.form['heartdisease'])
        
        ever_married_Yes = int(request.form['marriedstatus'])
        
        worktype =int( request.form.get('worktype'))
        
        
        # work_type_encoded_Govt_job=0
        work_type_Never_worked=0
        work_type_Private=0
        work_type_Self_employed=0
        work_type_children=0
        if(worktype==0):
            work_type_children=1
            print(work_type_children)
        # elif(worktype==1):
        #     work_type_encoded_Govt_job=1
            
        elif(worktype==2):
            work_type_Never_worked=1
            
        elif(worktype==3):
             work_type_Private=1
             
        elif(worktype==4):
            work_type_Self_employed=1
            
            
       

        Residence_type_Urban = int(request.form['residencetype'])
    
        avg_glucose_level = float(request.form['gloucoselevel'])
        
        bmi = float(request.form.get('bodymassindex'))
       
        smoking_status = int(request.form['smokingstatus'])
        print(smoking_status)
        #smoking_status_encoded_Unknown=0
        smoking_status_formerly_smoked =0
        smoking_status_never_smoked =0
        smoking_status_smokes =0
        if(smoking_status==0):
            smoking_status_formerly_smoked =1
            
        elif(smoking_status==1):
            smoking_status_never_smoked =1
            
        elif(smoking_status==2):
              smoking_status_smokes =1
              
        # elif(smoking_status==3):
        #      smoking_status_encoded_Unknown=1
        #      print(smoking_status_encoded_Unknown)
        
        
        
        input_features = [age	,avg_glucose_level,	bmi	,gender_Male,gender_Other,hypertension_1,	heart_disease_1,ever_married_Yes,	work_type_Never_worked,	work_type_Private,	work_type_Self_employed,	work_type_children	,Residence_type_Urban,	smoking_status_formerly_smoked,smoking_status_never_smoked	,smoking_status_smokes]

        features_value = [np.array(input_features)]
        features_name = ['age'	,'avg_glucose_level',	'bmi'	,'gender_Male'	,'gender_Other','hypertension_1',	'heart_disease_1','ever_married_Yes',	'work_type_Never_worked',	'work_type_Private',	'work_type_Self-employed',	'work_type_children'	,'Residence_type_Urban',	'smoking_status_formerly smoked','smoking_status_never smoked'	,'smoking_status_smokes']
        df = pd.DataFrame(features_value, columns=features_name)
        my_prediction = strokemodel.predict(df)
        print(my_prediction)

        return render_template('strokeresult.html', prediction=my_prediction)
    elif request.method == 'GET':
        return render_template('strokeprediction.html')
    else:
        return render_template('home.html')    
        
        





# if __name__ == "__main__":
#   #app.run(host='0.0.0.0', port='5000', debug=True)
#   app.run()


# app = Flask(__name__)

# @app.route('/')
# def home():
# 	return render_template('main.html')


# @app.route('/predict', methods=['GET','POST'])
# def predict():
#     if request.method == 'POST':

#         age = int(request.form['age'])
#         sex = request.form.get('sex')
#         cp = request.form.get('cp')
#         trestbps = int(request.form['trestbps'])
#         chol = int(request.form['chol'])
#         fbs = request.form.get('fbs')
#         restecg = int(request.form['restecg'])
#         thalach = int(request.form['thalach'])
#         exang = request.form.get('exang')
#         oldpeak = float(request.form['oldpeak'])
#         slope = request.form.get('slope')
#         ca = int(request.form['ca'])
#         thal = request.form.get('thal')
        
#         data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
#         my_prediction = model.predict(data)
        
#         return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run()

