import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app= application

# import ridge and standard scaler
ridge_model= pickle.load(open('models/Rg.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler (1).pkl','rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict():
    if request.method=="POST":
        Temperature= float(request.form.get('Temperature'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        scaled_data=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result= ridge_model.predict(scaled_data)
        return render_template('home.html',result=result[0])
    else:

        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
