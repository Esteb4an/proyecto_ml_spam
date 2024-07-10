import pandas as pd
import joblib
import os

# Cargar el modelo y el vectorizador
model_path = os.path.join(os.path.dirname(__file__), '../models/spam_model/spam_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), '../models/spam_model/vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Función de predicción
def predecir_spam(mensaje):
    X = vectorizer.transform([mensaje])
    prediccion = model.predict(X)
    return prediccion[0]

# Ejemplo de uso
if __name__ == "__main__":
    mensaje_prueba = "Felicidades! Has ganado un ticket gratis a las Bahamas. Haz clic aquí para reclamarlo."
    resultado = predecir_spam(mensaje_prueba)
    print(f'El mensaje "{mensaje_prueba}" es clasificado como: {resultado}')

