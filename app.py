from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load("model.pkl")



app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None  

    if request.method == "POST":
        pregnancies = float(request.form["Pregnancies"])
        glucose = float(request.form["Glucose"])
        bloodpressure = float(request.form["BloodPressure"])
        skin = float(request.form["SkinThickness"])
        insulin = float(request.form["Insulin"])
        bmi = float(request.form["BMI"])
        dpf = float(request.form["DiabetesPedigreeFunction"])
        age = float(request.form["Age"])

        
        dados = np.array([[pregnancies, glucose, bloodpressure, skin, insulin, bmi, dpf, age]])

       
        pred = model.predict(dados)[0]
        prediction = "Diabetes" if pred == 1 else "Sem Diabetes" 

   
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
