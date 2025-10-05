
#Tarea 2 Fundamentos de Data Science 

""""Una corredora de propiedades en Santiago quiere predecir el precio (en UF) de departamentos. 
Tienen los siguientes datos:
datos = {'Superficie_m2': [50, 70, 65, 90, 45], 'Num_Habitaciones': [1, 2, 2, 3, 1], 
'Distancia_Metro_km': [0.5, 1.2, 0.8, 0.2, 2.0], 'Precio_UF': [2500, 3800, 3500, 5200, 2100]} Construye un modelo de regresión lineal múltiple 
para predecir el 'Precio_UF' y evalúa su rendimiento."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


datos = {
    'Superficie_m2': [50, 70, 65, 90, 45],
    'Num_Habitaciones': [1, 2, 2, 3, 1],
    'Distancia_Metro_km': [0.5, 1.2, 0.8, 0.2, 2.0],
    'Precio_UF': [2500, 3800, 3500, 5200, 2100]
}

df = pd.DataFrame(datos)


x= df[['Superficie_m2', 'Num_Habitaciones', 'Distancia_Metro_km']] #Variables
y = df['Precio_UF'] 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42) #Division train/test

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)#predicciones

print("\nCoeficienes:\n ", modelo.coef_)
print("\nIntercepto:\n ", modelo.intercept_)
print("\nMSE: \n", mean_squared_error(y_test, y_pred))
print("\nR^2:\n ", r2_score(y_test, y_pred))
print("\nPredicciones (en UF):\n", y_pred)
print("\nValores reales (en UF):\n", list(y_test))

