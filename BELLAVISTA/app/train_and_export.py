import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# === Cargar datos ===
df = pd.read_excel("BELLAVISTA.xlsx")

# Drop de columnas no numéricas/no usadas
drop_cols = ["Distrito", "Vista al exterior"]
df = df.drop(columns=drop_cols, errors="ignore")

# Target y features
TARGET = "Precio en soles corrientes"
X = df.drop(columns=[TARGET], errors="ignore")
y = df[TARGET]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

    # Pipeline: imputación (media) + regresión lineal
pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
            ("model", LinearRegression())
            ])

pipe.fit(X_train, y_train)

            # Métricas rápidas
y_pred = pipe.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

            # === Guardar artefactos para la app web ===
            # 1) Pipeline completo
joblib.dump(pipe, "pipeline.pkl")

            # 2) Orden de columnas esperado
with open("features.json", "w", encoding="utf-8") as f:
	json.dump(list(X.columns), f, ensure_ascii=False, indent=2)

print("✅ Exportado: pipeline.pkl y features.json")
