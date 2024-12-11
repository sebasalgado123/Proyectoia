import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify
import joblib

def cargar_dataset():
    df = pd.read_csv(r'C:\Users\sabas\Downloads\IMDB Dataset.csv')
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df

def limpiar_texto(texto):
    texto = re.sub(r'\W', ' ', texto.lower())
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def preprocesar_datos(df):
    df['cleaned_review'] = df['review'].apply(limpiar_texto)
    return df

def preparar_datos(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_review']).toarray()
    y = df['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer

def entrenar_modelo(X_train, y_train):
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
    return modelo

def guardar_modelo(modelo, vectorizer):
    joblib.dump(modelo, 'modelo_clasificador.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
app = Flask(__name__)
modelo = None
vectorizer = None

def cargar_modelos():
    global modelo, vectorizer
    modelo = joblib.load('modelo_clasificador.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

@app.route('/clasificar', methods=['POST'])
def clasificar():
    datos = request.json
    texto = datos['texto']
    texto_vectorizado = vectorizer.transform([limpiar_texto(texto)])
    prediccion = modelo.predict(texto_vectorizado)
    return jsonify({'sentimiento': 'positivo' if prediccion[0] == 1 else 'negativo'})

if __name__ == '__main__':
    df = cargar_dataset()
    df = preprocesar_datos(df)
    X_train, X_test, y_train, y_test, vectorizer = preparar_datos(df)

    modelo = entrenar_modelo(X_train, y_train)

    evaluar_modelo(modelo, X_test, y_test)

    guardar_modelo(modelo, vectorizer)

    cargar_modelos()

    app.run(debug=True)