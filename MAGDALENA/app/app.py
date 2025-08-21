import os
import io
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, flash

# --- Config Flask ---
app = Flask(__name__)
app.secret_key = "cambia_esto_por_una_llave_segura"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB
app.config["UPLOAD_EXTENSIONS"] = [".xlsx", ".xls", ".csv"]
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# --- Cargar modelo y features ---
PIPELINE_PATH = "pipeline.pkl"
FEATURES_PATH  = "features.json"

if not os.path.exists(PIPELINE_PATH) or not os.path.exists(FEATURES_PATH):
    raise RuntimeError("Faltan pipeline.pkl y/o features.json. Ejecuta primero train_and_export.py")

pipeline = joblib.load(PIPELINE_PATH)
with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    FEATURES = json.load(f)

TARGET = "Precio en soles corrientes"  # para validar uploads

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte a numérico las columnas previstas; deja otras tal cual."""
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def align_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reordena/crea columnas según FEATURES, agregando NaN si faltan."""
    aligned = pd.DataFrame()
    for col in FEATURES:
        if col in df.columns:
            aligned[col] = df[col]
        else:
            aligned[col] = np.nan
    return aligned

@app.route("/", methods=["GET"])
def index():
    """
    Renderiza un formulario dinámico con las FEATURES y
    una sección para subir archivo (xlsx/csv).
    """
    return render_template("index.html", features=FEATURES)

@app.route("/predict_single", methods=["POST"])
def predict_single():
    """Predicción desde el formulario."""
    try:
        # Construir un dict con los valores del form
        values = {}
        for col in FEATURES:
            raw = request.form.get(col, "").strip()
            # vacíos -> NaN; numéricos -> float
            if raw == "":
                values[col] = np.nan
            else:
                try:
                    values[col] = float(raw)
                except ValueError:
                    # si no es convertible, márcalo como NaN
                    values[col] = np.nan

        X_new = pd.DataFrame([values], columns=FEATURES)
        y_hat = pipeline.predict(X_new)[0]
        flash(f"Predicción (Precio en soles corrientes): {y_hat:,.2f}", "success")
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"Error al predecir: {e}", "danger")
        return redirect(url_for("index"))

@app.route("/predict_file", methods=["POST"])
def predict_file():
    """Predicción por archivo (xlsx/csv). Devuelve un archivo con la columna 'Predicho'."""
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No subiste ningún archivo.", "warning")
            return redirect(url_for("index"))

        _, ext = os.path.splitext(file.filename.lower())
        if ext not in app.config["UPLOAD_EXTENSIONS"]:
            flash("Formato no permitido. Sube .xlsx, .xls o .csv", "warning")
            return redirect(url_for("index"))

        # Leer DataFrame
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        # Eliminar columnas no usadas si vienen (no hace daño, solo limpia)
        df = df.drop(columns=["Distrito", "Vista al exterior"], errors="ignore")

        # Si el archivo trae el target, lo ignoramos (para no romper X)
        df = df.drop(columns=[TARGET], errors="ignore")

        # Numérico + alinear columnas
        df = coerce_numeric(df)
        X_new = align_features(df)

        # Predecir
        y_hat = pipeline.predict(X_new)

        # Armar salida
        out = df.copy()
        out["Predicho"] = y_hat

        # Entregar como Excel en memoria
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name="Predicciones")
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name="predicciones.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        flash(f"Error al procesar archivo: {e}", "danger")
        return redirect(url_for("index"))

if __name__ == "__main__":
    # Debug para desarrollo
    app.run(host="0.0.0.0", port=9090, debug=True)

