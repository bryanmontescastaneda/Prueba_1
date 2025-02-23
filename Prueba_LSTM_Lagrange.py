import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.interpolate import lagrange
from sklearn.metrics import r2_score

# Fijar semilla para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# Leer los datos del archivo Excel
ruta = r'D:\Medicion de capacidades\Medicion_De_Capacidades\Prediccion a la medida\240863_ceros en los datos.xlsx'
df = pd.read_excel(ruta)

# Asegurarnos de que las columnas son correctas (Fecha y Tráfico)
df = df[['Fecha', 'Tráfico']] # Solo nos interesan estas dos columnas
df['Fecha'] = pd.to_datetime(df['Fecha']) # Convertir la columna Fecha a tipo datetime

# Establecer la columna 'Fecha' como índice
df.set_index('Fecha', inplace=True)

# Resamplear los datos a frecuencia diaria
df_resampled = df.resample('D').mean()

# Función para la interpolación de Lagrange
def lagrange_interpolation(x, y, xi):
    poly = lagrange(x, y)
    return poly(xi)

# Aplicar la interpolación de Lagrange
df_interpolado = df_resampled.copy()
df_interpolado['Tráfico'] = df_interpolado['Tráfico'].interpolate(method='polynomial', order=2)

# Preguntar al usuario si desea predecir 7 días o 4 semanas
# Se puede modificar esta línea para elegir entre 'dias' o 'semanas'
prediccion_tipo = 'semanas' # Cambia a 'dias' o 'semanas' según lo que quieras predecir

# Si elegimos predicción semanal, agrupar los datos por semana
if prediccion_tipo == 'semanas':
    df_interpolado = df_interpolado.resample('W').mean()  # Promedio semanal

# Restablecer el índice
df_interpolado.reset_index(inplace=True)

# Extraemos la columna de tráfico
temperaturas = df_interpolado['Tráfico'].values.reshape(-1, 1)

# Escalamos el tráfico
scaler = MinMaxScaler(feature_range=(0, 1))
temperaturas_scaled = scaler.fit_transform(temperaturas)

# Crear las secuencias de entrada y salida
X = []
y = []

# Ajustar las secuencias de entrada según si es para días o semanas
if prediccion_tipo == 'dias':
    # Para predecir 8 días, usamos 16 datos pasados
    for i in range(len(temperaturas_scaled) - 16 - 8):  # Asegúrate de que haya suficiente espacio para 8 días
        X.append(temperaturas_scaled[i:i+16])  # Secuencia de 16 días
        y.append(temperaturas_scaled[i+16])  # Los siguientes 8 días como salida
else:
    # Para predecir 4 semanas, usamos 8 semanas pasadas
    for i in range(len(temperaturas_scaled) - 8 - 4):  # Asegúrate de que haya suficiente espacio para 4 semanas
        X.append(temperaturas_scaled[i:i+8])  # Secuencia de 8 semanas
        y.append(temperaturas_scaled[i+8])  # Las siguientes 4 semanas como salida

X = np.array(X)
y = np.array(y)

# Dividir los datos en entrenamiento (80%), validación (10%) y prueba (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# Redimensionamos X para que sea adecuado para LSTM: (muestras, pasos de tiempo, características)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Creamos el modelo LSTM
model = Sequential()
model.add(LSTM(85, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(1)) # Solo una salida, el tráfico del siguiente día o semana

# Compilamos el modelo
model.compile(optimizer='adam', loss='mse')

# Early stopping para evitar sobreentrenamiento
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Entrenamos el modelo con validación
historia = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Graficamos la pérdida de entrenamiento y validación
plt.figure(figsize=(10, 6))
plt.plot(historia.history['loss'], label='Pérdida de entrenamiento')
plt.plot(historia.history['val_loss'], label='Pérdida de validación')
plt.legend()
plt.title('Pérdida de entrenamiento y validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.show()

# Calculamos el RMSE en el conjunto de validación
y_pred_val = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
# Cálculo del MAPE en validación
mape_val = np.mean(np.abs((y_val - y_pred_val) / y_val)) * 100

# Cálculo del R² en validación
r2_val = r2_score(y_val, y_pred_val)

# Visualizamos las predicciones y los valores reales en el conjunto de validación
plt.figure(figsize=(10, 6))
plt.plot(df_interpolado.index[-len(y_val):], y_val, label='Valores reales')
plt.plot(df_interpolado.index[-len(y_val):], y_pred_val, label='Predicciones', linestyle='--')
plt.legend()
plt.title('Predicciones vs Valores reales')
plt.xlabel('Fecha')
plt.ylabel('Tráfico')
plt.show()

# Calculamos el RMSE en el conjunto de prueba
y_pred_test = model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Cálculo del MAPE en prueba
mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

# Cálculo del R² en prueba
r2_test = r2_score(y_test, y_pred_test)

# Visualizamos las predicciones y los valores reales en el conjunto de prueba
plt.figure(figsize=(10, 6))
plt.plot(df_interpolado.index[-len(y_test):], y_test, label='Valores reales')
plt.plot(df_interpolado.index[-len(y_test):], y_pred_test, label='Predicciones', linestyle='--')
plt.legend()
plt.title('Predicciones vs Valores reales en prueba')
plt.xlabel('Fecha')
plt.ylabel('Tráfico')
plt.show()

# Realizamos la predicción para los próximos 8 días o 4 semanas
if prediccion_tipo == 'dias':
    entrada = temperaturas_scaled[-16:].reshape((1, 16, 1))  # Para días, usamos los últimos 16 datos
    fechas_prediccion = pd.date_range(df_interpolado['Fecha'].max() + pd.Timedelta(days=1), periods=60, freq='D')
else:
    entrada = temperaturas_scaled[-8:].reshape((1, 8, 1))  # Para semanas, usamos las últimas 8 semanas
    fechas_prediccion = pd.date_range(df_interpolado['Fecha'].max() + pd.Timedelta(weeks=1), periods=16, freq='W')

# Realizamos las predicciones
predicciones_futuras = []
entrada_futura = entrada

for _ in range(len(fechas_prediccion)):
    prediccion = model.predict(entrada_futura)
    prediccion_desescalada = scaler.inverse_transform(prediccion)
    prediccion_desescalada = np.maximum(prediccion_desescalada, 0)  # Evita valores negativos
    predicciones_futuras.append(prediccion_desescalada[0][0])  # Tomamos la primera predicción de las 4
    # Actualizamos la entrada para el siguiente día o semana
    entrada_futura = np.append(entrada_futura[:, 1:, :], prediccion.reshape(1, 1, 1), axis=1)

# Creamos un DataFrame con las predicciones y las fechas correspondientes
df_predicciones = pd.DataFrame({
    'Fecha': fechas_prediccion,
    'Predicción_Tráfico': predicciones_futuras
})

# Guardamos las predicciones en un nuevo archivo Excel
ruta_guardado = r'D:\Medicion de capacidades\Medicion_De_Capacidades\Prediccion a la medida\Prueba2112025.xlsx'
df_predicciones.to_excel(ruta_guardado, index=False)

print(f"Las predicciones para los próximos 7 {prediccion_tipo} se han guardado en {ruta_guardado}")
print("\n📊 Resultados de Evaluación 📊")
print(f"RMSE Validación: {rmse_val:.4f}, Prueba: {rmse_test:.4f}")
print(f"MAPE Validación: {mape_val:.2f}%, Prueba: {mape_test:.2f}%")
print(f"R² Validación: {r2_val:.4f}, Prueba: {r2_test:.4f}")