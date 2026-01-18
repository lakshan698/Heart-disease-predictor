from flask import Flask, render_template, request
import numpy as np
import joblib

# Load your model
model = joblib.load('heart_risk_predict_regression_model.sav')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('patient_details.html')

@app.route('/getresults', methods=['POST'])
def getresults():
    result = request.form

    name = result['name']
    gender = float(result['gender'])
    age = float(result['age'])
    tc = float(result['tc'])
    hdl = float(result['hdl'])
    smoke = float(result['smoke'])
    
    # CORRECTED: Changed 'bpm' (Heart Rate) to 'bpmed' (BP Medication)
    # CORRECTED: Removed 'sbp' as it is not in the dataset image
    bpmed = float(result['bpmed']) 
    
    diab = float(result['diab'])

    # The array must match the dataset order: 
    # SEX, AGEIR, TC, HDL, SMOKE_, BPMED, DIAB_01
    test_data = np.array([gender, age, tc, hdl, smoke, bpmed, diab]).reshape(1, -1)

    prediction = model.predict(test_data)

    # --- ADD THIS LOGIC ---
    risk_value = prediction[0][0]
    
    # If risk is negative, make it 0
    if risk_value < 0:
        risk_value = 0
    # ----------------------
    
    resultDict = {"name": name, "risk": round(risk_value, 2)}

    return render_template('patient_results.html', results=resultDict)

if __name__ == '__main__':
    app.run(debug=True)