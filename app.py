from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

with open('model/bike_model_rscv.pkl', 'rb') as pickle_file:
    model = pickle.load(pickle_file)

app = Flask(__name__) 

@app.route('/')
def index():
    return render_template('home.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    t1 = request.form['a']
    t2 = request.form['b']
    hum= request.form['c']
    wind_speed = request.form['d']
    weather_code = request.form['e']
    is_holiday = request.form['f']
    is_weekend = request.form['g']
    season = request.form['h']
    month = request.form['i']
    day = request.form['j']
    hour = request.form['k']

    input_data = pd.DataFrame([[t1,t2,hum,wind_speed,weather_code,is_holiday,is_weekend,season,month,day,hour]],
                            columns=['t1','t2','hum','wind_speed','weather_code','is_holiday','is_weekend','season','month','day','hour'],
                            dtype=float)
    prediction = model.predict(input_data)[0]

    return render_template('after.html', result=np.round(prediction))

if __name__ == "__main__":
    app.run(debug=True)
