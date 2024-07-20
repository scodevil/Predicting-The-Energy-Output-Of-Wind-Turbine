from flask import Flask, render_template, request
import joblib
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import requests
from pprint import pprint






ct = pickle.load(open("power_prediction.pkl","rb"))
sc = pickle.load(open("sc.pkl","rb"))

app = Flask(__name__)
@app.route('/')
def loadpage():
    return render_template("page1.html")
@app.route('/index')
def indexpage():
    return render_template("index.html")

@app.route('/y_predict', methods = ["POST"])
def prediction():
    
    city = request.form["City"]
    
    url = 'https://api.openweathermap.org/data/2.5/weather?q={}&appid=7ccb39f016ab18098ad4cdfbf97d04b6&units=metric'.format(city)

    res = requests.get(url)

    data = res.json()

    temp = data['main']['temp']
    wind_speed = data['wind']['speed']
    
    Day = request.form["Day"]
    Month = request.form["Month"]
    Hour = request.form["Hour"]
    Theoretical_Power_Curve = request.form["Theoretical_Power_Curve"]
    
    x_test = [[float(Theoretical_Power_Curve),float(wind_speed),int(Month),int(Day),int(Hour)]]
    print(x_test)
    
    p = np.array(sc.transform(x_test))

    
    prediction =ct.predict(p)
    
        
   
    return render_template("index.html",prediction_text = prediction)

    
    
if __name__ == "__main__":
    app.run(debug = False)


