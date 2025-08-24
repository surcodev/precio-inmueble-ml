from flask import Flask, render_template, request
import joblib
import pandas as pd
import time, random

app = Flask(__name__)

# Diccionario de modelos por distrito
MODELOS = {
    "Bellavista": "dataset_bellavista.pkl",
    "Los Olivos": "dataset_los_olivos.pkl",
    "Magdalena": "dataset_magdalena.pkl"
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None
    error = None

    if request.method == "POST":
        try:
            distrito = request.form["distrito"]
            modelo_path = MODELOS[distrito]

            # Cargar modelo
            model = joblib.load(modelo_path)

            # 👇 Usa los nombres EXACTOS del entrenamiento
            data = {
                "Año": int(request.form["anio"]),
                "Trimestre": int(request.form["trimestre"]),
                "Tipo de cambio": float(request.form["tipo_cambio"]),
                "Superficie ": float(request.form["superficie"]),  # 👈 OJO espacio al final
                "Número de habitaciones": int(request.form["habitaciones"]),
                "Número de baños": int(request.form["banos"]),
                "Número de garajes": int(request.form["garajes"]),
                "Piso de ubicación": float(request.form["piso"]),
                "Años de antigüedad": int(request.form["antiguedad"]),
            }

            df = pd.DataFrame([data])

            # Simular demora 1–3 seg
            time.sleep(random.randint(1, 3))

            # Predicción
            prediccion = model.predict(df)[0]

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediccion=prediccion, distritos=MODELOS.keys(), error=error)


if __name__ == "__main__":
    app.run(debug=True)
