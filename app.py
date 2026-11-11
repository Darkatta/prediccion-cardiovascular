import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils import (COLUMNAS, ETIQUETAS, RANGOS, construir_registro, predecir,
                   validar_entrada)

BASE = Path(__file__).parent

st.set_page_config(page_title="Predicción de enfermedad cardiovascular",
                   page_icon="❤️", layout="wide")


# ---------- Carga de archivos (el modelo NO se reentrena) ----------
@st.cache_resource
def cargar_modelo():
    return joblib.load(BASE / "modelo_cardiovascular.joblib")


@st.cache_data
def cargar_datos():
    return pd.read_csv(BASE / "Heart_disease_cleveland_new.csv")


@st.cache_data
def cargar_metricas():
    return json.loads((BASE / "metricas.json").read_text(encoding="utf-8"))


@st.cache_data
def cargar_prueba():
    return pd.read_csv(BASE / "datos_prueba.csv", index_col=0)


try:
    modelo = cargar_modelo()
    df = cargar_datos()
    metricas = cargar_metricas()
    prueba = cargar_prueba()
except FileNotFoundError as e:
    st.error(f"Falta un archivo en el repositorio: {e.filename}")
    st.stop()

# ---------- Encabezado ----------
st.title("❤️ Predicción de enfermedad cardiovascular")
st.caption("Minería de Datos · Universidad de la Costa (CUC) · Tercer corte")
st.warning("**Prototipo académico.** Esta aplicación no sustituye el diagnóstico "
           "ni la decisión de un profesional de la salud.")

tab_proy, tab_eda, tab_pred, tab_met = st.tabs(
    ["📌 Proyecto", "📊 Exploración de datos", "🩺 Predicción", "📈 Métricas y limitaciones"])

# ---------- 1. Proyecto ----------
with tab_proy:
    st.subheader("Objetivo")
    st.write("Clasificar la presencia o ausencia de enfermedad cardiovascular a partir "
             "de información clínica y demográfica de un paciente, usando técnicas de "
             "minería de datos y aprendizaje automático.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Integrantes")
        st.markdown("- Ronaldo Cruz\n- Santiago Ayubb\n- Orlando Duran\n- Janier Ramírez")
        st.caption("Docente: José Escorcia-Gutiérrez")
    with c2:
        st.subheader("Fuente de los datos")
        st.write("**Cleveland Heart Disease Dataset** (UCI Machine Learning Repository): "
                 f"{df.shape[0]} pacientes y {df.shape[1] - 1} variables predictoras. "
                 "Variable objetivo `target`: 0 = ausencia, 1 = presencia.")
    st.subheader("Modelo")
    st.write(f"**{metricas['modelo_final']}** dentro de un pipeline con escalado de variables "
             "numéricas y codificación one-hot de las categóricas. Se eligió frente al árbol de "
             "decisión por su mayor recall en la clase *presencia*, menos falsos negativos y "
             "mejor F1 en validación cruzada (no solo por accuracy).")

# ---------- 2. EDA ----------
with tab_eda:
    st.subheader("Visualizaciones del análisis exploratorio")
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        conteo = df["target"].value_counts().sort_index()
        ax.bar(["Ausencia (0)", "Presencia (1)"], conteo.values, color=["#4c9f70", "#d9534f"])
        for i, v in enumerate(conteo.values):
            ax.text(i, v + 2, f"{v} ({v / len(df):.0%})", ha="center")
        ax.set_ylabel("Pacientes")
        ax.set_title("Distribución del diagnóstico")
        st.pyplot(fig)
        st.caption("Clases casi balanceadas: no se requirió balanceo.")

    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist([df.loc[df.target == 0, "age"], df.loc[df.target == 1, "age"]],
                bins=10, stacked=True, color=["#4c9f70", "#d9534f"],
                label=["Ausencia", "Presencia"])
        ax.set_xlabel("Edad (años)")
        ax.set_ylabel("Pacientes")
        ax.set_title("Edad según diagnóstico")
        ax.legend()
        st.pyplot(fig)
        st.caption("Los pacientes con enfermedad tienden a ser de mayor edad.")

    corr = df.corr()["target"].drop("target").sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(corr.index, corr.values, color=np.where(corr.values >= 0, "#d9534f", "#4c9f70"))
    ax.set_xlabel("Correlación con target")
    ax.set_title("Correlación de cada variable con el diagnóstico")
    st.pyplot(fig)
    st.caption("La correlación mide asociación lineal, no causalidad.")

# ---------- 3. Predicción ----------
with tab_pred:
    st.subheader("Ingreso de datos del paciente")
    st.write("Completa los valores y presiona **Predecir**. Se validan rangos y categorías "
             "antes de usar el modelo.")

    thal_opts = sorted(int(v) for v in df["thal"].unique())
    cats_validas = {
        "sex": ETIQUETAS["sex"], "cp": ETIQUETAS["cp"], "fbs": ETIQUETAS["fbs"],
        "restecg": ETIQUETAS["restecg"], "exang": ETIQUETAS["exang"],
        "slope": ETIQUETAS["slope"], "ca": ETIQUETAS["ca"], "thal": thal_opts,
    }

    def sel(col, titulo, opciones, defecto=0):
        return st.selectbox(titulo, list(opciones), index=defecto,
                            format_func=lambda v: ETIQUETAS[col].get(v, f"Código {v}"))

    with st.form("form_prediccion"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Edad (años)", 0, 150, 55, 1)
            sex = sel("sex", "Sexo", ETIQUETAS["sex"], 1)
            cp = sel("cp", "Tipo de dolor en el pecho", ETIQUETAS["cp"])
            trestbps = st.number_input("Presión arterial en reposo (mm Hg)", 0, 400, 130, 1)
            chol = st.number_input("Colesterol sérico (mg/dl)", 0, 1000, 240, 1)
        with c2:
            fbs = sel("fbs", "Azúcar en ayunas > 120 mg/dl", ETIQUETAS["fbs"])
            restecg = sel("restecg", "Electrocardiograma en reposo", ETIQUETAS["restecg"])
            thalach = st.number_input("Frecuencia cardíaca máxima (lpm)", 0, 400, 150, 1)
            exang = sel("exang", "Angina inducida por ejercicio", ETIQUETAS["exang"])
            oldpeak = st.number_input("Depresión del ST (mm)", 0.0, 20.0, 1.0, 0.1, format="%.1f")
        with c3:
            slope = sel("slope", "Pendiente del segmento ST", ETIQUETAS["slope"], 1)
            ca = sel("ca", "Vasos principales coloreados", ETIQUETAS["ca"])
            thal = sel("thal", "Resultado de talasemia (thal)", thal_opts)
        enviar = st.form_submit_button("Predecir", type="primary")

    if enviar:
        valores = {"age": int(age), "sex": sex, "cp": cp, "trestbps": int(trestbps),
                   "chol": int(chol), "fbs": fbs, "restecg": restecg,
                   "thalach": int(thalach), "exang": exang, "oldpeak": float(oldpeak),
                   "slope": slope, "ca": ca, "thal": thal}
        errores = validar_entrada(valores, cats_validas)
        if errores:
            st.error("No se puede predecir. Corrige estos datos:")
            for e in errores:
                st.write(f"- {e}")
        else:
            clase, prob = predecir(modelo, construir_registro(valores))
            r1, r2 = st.columns(2)
            r1.metric("Clasificación del modelo",
                      "Presencia de enfermedad" if clase == 1 else "Ausencia de enfermedad")
            r2.metric("Probabilidad estimada de presencia", f"{prob:.1%}")
            st.progress(prob)
            st.caption("Umbral de clasificación: 50 %. Resultado orientativo de un prototipo "
                       "académico; consulte a un profesional de la salud.")

    with st.expander("Verificar con un registro del conjunto de prueba"):
        idx = st.selectbox("Índice del registro", list(prueba.index))
        fila = prueba.loc[[idx]]
        clase, prob = predecir(modelo, fila[COLUMNAS])
        st.dataframe(fila.T, use_container_width=True)
        st.write(f"**Valor real:** {int(fila['target'].iloc[0])} · "
                 f"**Predicción:** {clase} · **P(presencia):** {prob:.4f}")
        st.caption("Debe coincidir con la predicción del notebook para el mismo registro.")

# ---------- 4. Métricas y limitaciones ----------
with tab_met:
    final = metricas["modelos"][metricas["modelo_final"]]
    st.subheader(f"Desempeño del modelo final: {metricas['modelo_final']}")
    st.caption(f"Entrenamiento: {metricas['n_train']} pacientes · "
               f"Prueba: {metricas['n_test']} pacientes (división estratificada 80/20).")
    m = st.columns(5)
    m[0].metric("Accuracy", f"{final['accuracy']:.3f}")
    m[1].metric("Precision", f"{final['precision']:.3f}")
    m[2].metric("Recall", f"{final['recall']:.3f}")
    m[3].metric("F1", f"{final['f1']:.3f}")
    m[4].metric("F1 validación cruzada", f"{final['cv_f1_mean']:.3f} ± {final['cv_f1_std']:.3f}")
    st.caption("Precision, recall y F1 corresponden a la clase *presencia*.")

    c1, c2 = st.columns([1, 1.2])
    with c1:
        cm = np.array(final["matriz_confusion"])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(cm, cmap="Blues")
        etiquetas = ["Ausencia", "Presencia"]
        ax.set_xticks([0, 1], etiquetas)
        ax.set_yticks([0, 1], etiquetas)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de confusión")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        st.pyplot(fig)
        st.caption(f"Falsos negativos: {cm[1, 0]} · Falsos positivos: {cm[0, 1]}")
    with c2:
        tabla = pd.DataFrame({
            nombre: {"Accuracy": d["accuracy"], "Precision": d["precision"],
                     "Recall": d["recall"], "F1": d["f1"],
                     "F1 CV (media)": d["cv_f1_mean"], "F1 CV (desv.)": d["cv_f1_std"],
                     "Falsos negativos": d["matriz_confusion"][1][0]}
            for nombre, d in metricas["modelos"].items()}).round(3)
        st.write("**Comparación de modelos**")
        st.dataframe(tabla, use_container_width=True)

    st.subheader("Limitaciones")
    st.markdown(
        "- Dataset pequeño (303 pacientes) y conjunto de prueba de solo 61: las métricas "
        "pueden cambiar con otra división.\n"
        "- No se optimizaron hiperparámetros.\n"
        "- `ca` y `thal` provienen de pruebas diagnósticas adicionales, por lo que el modelo "
        "no equivale a una evaluación con datos básicos del paciente.\n"
        "- El modelo aún comete falsos negativos, el error más grave en un contexto de tamizaje.\n"
        "- Los datos provienen de una sola institución; no se garantiza que generalicen a "
        "otras poblaciones.\n"
        "- Los resultados son asociaciones estadísticas, no relaciones causales.")
