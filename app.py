from flask import Flask, render_template, request
import numpy as np
import json
import os
import joblib

#pip install scikit-learn

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None  

    if request.method == "POST":
        name = request.form["Name"]
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


        novo_registro = {
            "Nome": name,
            "Gravidezes": pregnancies,
            "Glicose": glucose,
            "Pressão Sanguínea": bloodpressure,
            "Espessura da Pele": skin,
            "Insulina": insulin,
            "BMI": bmi,
            "DPF": dpf,
            "Idade": age,
            "Resultado": prediction
        }

        if not os.path.exists("dados.json"):
            with open("dados.json", "w") as f:
                json.dump([], f)

        with open("dados.json", "r") as f:
            data = json.load(f)

        data.append(novo_registro)

        with open("dados.json", "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

   
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
