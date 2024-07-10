import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os

# Cargar el dataset
ruta_datos = os.path.join(os.path.dirname(__file__), '../data/spam_dataset.csv')
df = pd.read_csv(ruta_datos)

# Verificar las columnas del DataFrame
print("Columnas del DataFrame:", df.columns)

# Preprocesamiento de datos
X = df['message']
y = df['label']

# Vectorización de texto
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# División de datos en entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
modelo = MultinomialNB()
modelo.fit(X_entrenamiento, y_entrenamiento)

# Evaluación del modelo
y_prediccion = modelo.predict(X_prueba)
precision = accuracy_score(y_prueba, y_prediccion)
print(f'Precisión del modelo: {precision * 100:.2f}%')

# Guardar el modelo y el vectorizador
ruta_modelo = os.path.join(os.path.dirname(__file__), '../models/spam_model/spam_model.pkl')
ruta_vectorizador = os.path.join(os.path.dirname(__file__), '../models/spam_model/vectorizer.pkl')

joblib.dump(modelo, ruta_modelo)
joblib.dump(vectorizer, ruta_vectorizador)



