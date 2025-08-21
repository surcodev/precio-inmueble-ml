import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_excel("BELLAVISTA.xlsx")
filas, columnas = df.shape
print("Número de filas:", filas)
print("Número de columnas:", columnas)
df.head()
df.drop(["Distrito", "Vista al exterior"], axis=1, inplace=True)
corr = df.corr()
plt.figure(figsize=(len(corr.columns) * 0.7, len(corr.columns) * 0.7))  # ajustar tamaño dinámico
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    cbar=True,
    square=True,        # celdas cuadradas
    linewidths=0.5
)
plt.title("Matriz de Correlación", fontsize=16)
plt.xticks(rotation=90, fontsize=8)   # gira labels eje X
plt.yticks(rotation=0, fontsize=8)    # labels eje Y
plt.tight_layout()
plt.show()
df.head()
df.info()
df.head()
df.isnull().sum()
df = df.fillna(df.mean(numeric_only=True))
df.head()
df.info()
X = df.drop(columns=["Precio en soles corrientes"])
y = df["Precio en soles corrientes"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred
import pandas as pd
resultados = pd.DataFrame({
    "Real": y_test.values,
    "Predicho": y_pred,
    "Diferencia": y_test.values - y_pred
})
print(resultados)  # mostrar las primeras filas
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # línea ideal
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Regresión Lineal Múltiple: Real vs Predicho")
plt.show()
residuos = y_test - y_pred
plt.figure(figsize=(6,6))
plt.scatter(y_pred, residuos, alpha=0.7)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Valores predichos")
plt.ylabel("Residuos (Real - Predicho)")
plt.title("Gráfico de Residuos")
plt.show()
coeficientes = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": model.coef_
})
print(coeficientes.sort_values(by="Coeficiente", ascending=False))
import joblib
joblib.dump(model, "modelo.pkl")
get_ipython().system('ls')
modelo_cargado = joblib.load("modelo.pkl")
y_pred = modelo_cargado.predict(X_test)
y_pred
X_test
df_nuevo = pd.read_excel("b1.xlsx")
df_nuevo.drop(["Distrito", "Vista al exterior"], axis=1, inplace=True)
df_nuevo = df_nuevo.fillna(df_nuevo.mean(numeric_only=True))
X_new = df_nuevo.drop(columns=["Precio en soles corrientes"], errors="ignore")
y_pred = modelo_cargado.predict(X_new)
y_pred
