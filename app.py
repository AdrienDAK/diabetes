from flask import Flask,request,jsonify

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('entry_form.html')

@app.route('/result/', methods=['GET','POST'])
def predict():
    
    if request.method == "POST":
        #get form data
        nbre_pregnant = int(request.form.get('nbre_pregnant'))
        plasma = int(request.form.get('plasma'))
        pressure = float(request.form.get('pressure'))
        thickness = float(request.form.get('thickness'))
        insulin = float(request.form.get('insulin'))
        mass_index = float(request.form.get('mass_index'))
        pedigree = float(request.form.get('pedigree'))
        age = int(request.form.get('age'))
        
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(nbre_pregnant,plasma,pressure,thickness,insulin,mass_index,pedigree,age)            #pass prediction to template
            return render_template('result.html', prediction = prediction)
   
        except ValueError:
            return "Please Enter valid values"
  
        pass
    pass

def preprocessDataAndPredict(nbre_pregnant,plasma,pressure,thickness,insulin,mass_index,pedigree,age):
    
    #keep all inputs in array
    test_data = pd.DataFrame()

    medians_ = {'Number pregnant':3, 'Plasma':117, 'pressure':72, 'thickness':23, 'insulin':30,
                   'mass index':32, 'Diabetes pedigree function':0.3725, 'Age':29}

    test_data['Number pregnant'] = [np.log(nbre_pregnant+1)]

    if plasma==0 or plasma>200: test_data['Plasma'] = [medians_['Plasma']]
    else: test_data['Plasma'] = [plasma]

    if pressure<35: test_data['pressure'] = [medians_['pressure']]
    else: test_data['pressure'] = [pressure]

    if thickness ==0 or thickness>80: test_data['thickness'] = [medians_['thickness']]
    else: test_data['thickness'] = [thickness]

    if insulin<10 or insulin>50: test_data['insulin'] = [np.log(medians_['insulin']+1)]
    else: test_data['insulin'] = [np.log(insulin+1)]

    if mass_index==0 or mass_index>50: test_data['mass index'] = [medians_['mass index']]
    else: test_data['mass index'] = [mass_index]

    if pedigree<0.078 or pedigree>2.42: test_data['Diabetes pedigree function'] = [np.log(medians_['Diabetes pedigree function'])]
    else: test_data['Diabetes pedigree function'] = [np.log(pedigree)]

    if age<21 or age>81: test_data['Age'] = [np.log(medians_['Age'])]
    else: test_data['Age'] = [np.log(age)]


    #open file
    file = open('scaler.pkl','rb')

    #load scaler
    scaler = joblib.load(file)

    #Scale the data
    test_data = scaler.transform(test_data)
    
    #open file
    file = open("rfc.pkl","rb")
    
    #load trained model
    model = joblib.load(file)
    
    #predict
    prediction = model.predict(test_data)
    
    return prediction
    
    pass

if __name__ == '__main__':
    app.run(debug=True)