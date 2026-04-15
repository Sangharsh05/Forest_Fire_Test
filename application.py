import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regressor and standard scaler pickle
ridge_model=pickle.load(open("models/ridge.pkl","rb"))
standard_scaler=pickle.load(open("models/scaler.pkl","rb"))





@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":

        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        # Scale input
        new_data_scaled = standard_scaler.transform(
            [[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]]
        )

        # Prediction
        prediction = ridge_model.predict(new_data_scaled)[0]

        # Convert numeric → label
        if prediction >= 0.5:
            result = "High Fire Risk"
        else:
            result = "Low Fire Risk"

        return render_template('home.html', results=result)

    else: 
        return render_template('home.html')


if __name__=="__main__":
    app.run(debug=True)