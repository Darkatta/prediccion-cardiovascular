"""Lógica de la app sin Streamlit (validación, construcción del registro y predicción)."""
import math

import pandas as pd

# Orden de las 13 variables predictoras (las mismas del notebook)
COLUMNAS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Variables numéricas: (mínimo, máximo, nombre, unidad). Rango clínicamente plausible.
RANGOS = {
    'age': (18, 100, 'Edad', 'años'),
    'trestbps': (80, 220, 'Presión arterial en reposo', 'mm Hg'),
    'chol': (100, 600, 'Colesterol sérico', 'mg/dl'),
    'thalach': (60, 220, 'Frecuencia cardíaca máxima', 'lpm'),
    'oldpeak': (0.0, 7.0, 'Depresión del ST inducida por ejercicio', 'mm'),
}

# Variables categóricas: código -> etiqueta comprensible
# OJO: verifica slope y thal con la tabla de variables de tu notebook.
ETIQUETAS = {
    'sex': {0: 'Femenino', 1: 'Masculino'},
    'cp': {0: 'Angina típica', 1: 'Angina atípica',
           2: 'Dolor no anginoso', 3: 'Asintomático'},
    'fbs': {0: 'No (≤ 120 mg/dl)', 1: 'Sí (> 120 mg/dl)'},
    'restecg': {0: 'Normal', 1: 'Anomalía de la onda ST-T',
                2: 'Hipertrofia ventricular izquierda'},
    'exang': {0: 'No', 1: 'Sí'},
    'slope': {0: 'Ascendente', 1: 'Plana', 2: 'Descendente'},
    'ca': {0: '0 vasos', 1: '1 vaso', 2: '2 vasos', 3: '3 vasos'},
    'thal': {},  # si lo llenas {codigo: 'descripción'}; si no, se muestra "Código n"
}


def validar_entrada(valores, categorias_validas):
    """Devuelve una lista de mensajes de error (vacía si todo es válido)."""
    errores = []
    for col, (minimo, maximo, nombre, unidad) in RANGOS.items():
        v = valores.get(col)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (
                isinstance(v, float) and math.isnan(v)):
            errores.append(f'{nombre}: debe ser un número.')
        elif not (minimo <= v <= maximo):
            errores.append(f'{nombre}: {v} {unidad} está fuera del rango '
                           f'permitido ({minimo} a {maximo} {unidad}).')
    for col, permitidas in categorias_validas.items():
        if valores.get(col) not in set(permitidas):
            errores.append(f'{col}: categoría no válida ({valores.get(col)!r}).')
    return errores


def construir_registro(valores):
    """DataFrame de una fila con las columnas en el orden del entrenamiento."""
    return pd.DataFrame([valores])[COLUMNAS]


def predecir(modelo, registro):
    """Devuelve (clase predicha, probabilidad de presencia)."""
    idx = list(modelo.classes_).index(1)
    prob = float(modelo.predict_proba(registro)[0][idx])
    clase = int(modelo.predict(registro)[0])
    return clase, prob
