# Autor Enzo Luciano Iriarte
# Tema: Clasificador NLP de tipos de habitaciones segun tipos de departamentos. Sub proyecto de un ChatBot
# Fecha: 8/04/2023

import pandas as pd
import numpy as np
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
import joblib

# Cargo los datos de Airbnb
df = pd.read_csv('listings.csv')
print(df.shape)
print(df.columns)
print(df.info)

print(f"Hay datos ausentes? {df.isnull().any().any()}")

# Analizamos el dataset

print(df.room_type.unique())

import re
def preprocesar_texto(texto):
    if isinstance(texto, str):
        # eliminar etiquetas HTML
        texto = re.sub(r'<[^>]+>', '', texto)
        # eliminar caracteres que no sean texto o números
        texto = re.sub(r'\W', ' ', texto)
        # convertir a minúsculas
        texto = texto.lower()
    return texto

for columna in df.columns:
    df[columna] = df[columna].apply(preprocesar_texto)

df= df[['property_type', 'room_type']]

df = df.dropna(subset=['property_type'])
df = df.dropna(subset=['room_type'])
print(df.info())
print(df.isna().sum() == 0)

df = df.sample(frac=1).reset_index(drop=True)
#print(df.head(30))
df.to_csv('df.csv', sep=";",index=False)
print(f"Hay datos ausentes? {df.isnull().any().any()}")

import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords

def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

df['property_type'] = df['property_type'].apply(remove_stopwords)

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lematizacion(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

df["property_type"] = df['property_type'].apply(lematizacion)

# Divido los datos en conjuntos de entrenamiento y prueba
X_train_full, X_test, y_train_full, y_test = train_test_split(df['property_type'], df['room_type'], test_size=0.2, random_state=42)

#Guardo X_train_full en un archivo csv
X_train_full.to_csv('X_train_full.csv', index=False)

# Creo un objeto vectorizador TF-IDF
tfidf = TfidfVectorizer()

batch_size = 5000
num_batches = len(X_train_full) // batch_size + 1

from sklearn.naive_bayes import MultinomialNB

models = []

#Entreno el modelo Bayesiano Naive en lotes
for i in range(num_batches):
    start = i * batch_size
    end = (i + 1) * batch_size
    X_train_batch = X_train_full[start:end]
    y_train_batch = y_train_full[start:end]
    
    # Ajusto el vectorizador a los datos de entrenamiento
    tfidf.fit(X_train_batch)

    # Transformo los datos de entrenamiento y prueba en vectores TF-IDF
    X_train_batch_tfidf = tfidf.transform(X_train_batch)

    # Entreno el modelo Bayesiano Naive en el lote actual
    mnb = MultinomialNB()
    mnb.fit(X_train_batch_tfidf, y_train_batch)
    models.append(mnb)

#Combino los modelos entrenados en una sola instancia del modelo
mnb = MultinomialNB()
mnb.estimators_ = models

#Ajusto el modelo al conjunto completo de datos de entrenamiento
mnb.fit(tfidf.transform(X_train_full), y_train_full)

#Evaluo el modelo en los datos de prueba
X_test_tfidf = tfidf.transform(X_test)

y_pred = mnb.predict(X_test_tfidf)



#Calculo las métricas de desempeño
print('Exactitud:', accuracy_score(y_test, y_pred))
print('Precisión:', precision_score(y_test, y_pred, average='weighted'))
print('Recall:', recall_score(y_test, y_pred, average='weighted'))
print('Puntuación F1:', f1_score(y_test, y_pred, average='weighted'))

#Texto de prueba
texto = "Hello, I want to rent an apartment with two rooms."
texto_transformado = tfidf.transform([texto])
y_pred = mnb.predict(texto_transformado)

print(y_pred)

joblib.dump(mnb, 'modelo_mnb.joblib')