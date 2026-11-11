# Predicción de enfermedad cardiovascular — Despliegue en Streamlit

Proyecto de aula de **Minería de Datos** (Universidad de la Costa, CUC) — Tercer corte.

**Aplicación en línea:** https://prediccion-cardiovascular.streamlit.app/

> ⚠️ Prototipo académico. No sustituye el diagnóstico ni la decisión de un profesional de la salud.

## Integrantes
- Ronaldo Cruz
- Santiago Ayubb
- Orlando Duran
- Janier Ramírez

Docente: José Escorcia-Gutiérrez

## Descripción
Clasificación de la presencia (1) o ausencia (0) de enfermedad cardiovascular con el
**Cleveland Heart Disease Dataset** (303 pacientes, 13 variables predictoras). El modelo final es
una **regresión logística** dentro de un pipeline (escalado de numéricas + one-hot de categóricas).

## Resultados (conjunto de prueba, 61 pacientes)
| Modelo | Accuracy | Recall (presencia) | F1 (presencia) | Falsos negativos | F1 CV (media ± desv.) |
|---|---|---|---|---|---|
| Regresión logística (final) | 0.869 | 0.929 | 0.867 | 2 | 0.827 ± 0.043 |
| Árbol de decisión (prof. 4) | 0.787 | 0.821 | 0.780 | 5 | 0.720 ± 0.089 |

Se eligió la regresión logística por recall, falsos negativos y estabilidad en validación cruzada,
no solo por accuracy. Las métricas superan 0,70, así que no fue necesario corregir el pipeline;
se revisó con la ficha de calidad (datos, separación antes de transformar, pipeline, evaluación).

## Estructura del repositorio
| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación Streamlit |
| `utils.py` | Validación de entradas y predicción |
| `modelo_cardiovascular.joblib` | Pipeline completo (preprocesamiento + modelo) |
| `metricas.json` | Métricas y matrices de confusión generadas en el notebook |
| `datos_prueba.csv` | Conjunto de prueba, para verificar registros |
| `Heart_disease_cleveland_new.csv` | Datos para el EDA |
| `Analisis_Exploratorio_Cardiovascular.ipynb` | Notebook actualizado |
| `requirements.txt` | Dependencias |

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pruebas realizadas
| Prueba | Resultado |
|---|---|
| Caso normal | Edad 55, sexo masculino, dolor de pecho angina típica, presión 130 mm Hg, colesterol 240 mg/dl, azúcar en ayunas No, ECG normal, frecuencia máxima 150 lpm, angina por ejercicio No, depresión ST 1,0, pendiente plana, 0 vasos, thal código 1. La app clasificó **Ausencia de enfermedad** con probabilidad de presencia de **14,3 %**. |
| Entrada inválida | Edad = 150. La app no predijo y mostró: "Edad: 150 años está fuera del rango permitido (18 a 100 años)." |
| Registro conocido del conjunto de prueba (índice 67) | Notebook: P(presencia) = 0,3395 · Streamlit: P(presencia) = 0,3395 · valor real 0, predicción 0. Coinciden. |

## Limitaciones
Dataset pequeño (303) y prueba de 61 pacientes; sin optimización de hiperparámetros; `ca` y `thal`
vienen de pruebas diagnósticas adicionales; persisten falsos negativos; datos de una sola institución.


## Uso de inteligencia artificial
**Herramienta:** Claude (Anthropic)
**Propósito:** generar la estructura de la aplicación Streamlit (app.py y utils.py), la validación de entradas, la celda del notebook para guardar el pipeline y las métricas, y orientación para el despliegue en GitHub y Streamlit Cloud.
**Prompt principal:** Se adjuntó el enunciado de la actividad (PDF "Despliegue en Streamlit") junto con el anuncio del docente de inicio del tercer corte, y se escribió: "Miralo q tengo q hacer". Después se adjuntaron la ficha de revisión de calidad del modelado y el tutorial de Streamlit ("estos dos archivos los mandó el profesor, ya te mando mi notebook y lo q un compañero adelantó"). Por último se adjuntó el notebook del segundo avance y se pidió: "ese es el notebook, ayúdame desde ahí".
**Cómo se verificó la respuesta:** se ejecutó la celda en el notebook y se compararon las métricas de la app con las del notebook; se probó un caso normal, una entrada inválida y un registro del conjunto de prueba (índice 67), cuya probabilidad coincidió en el notebook y en Streamlit.
**Problemas encontrados:** el primer despliegue se quedó instalando dependencias porque Streamlit Cloud usó Python 3.14 y las versiones fijadas en requirements.txt no tenían instaladores para esa versión. Además, en la tarjeta de F1 de validación cruzada el texto se cortaba en pantallas estrechas.
**Ajustes realizados:** se eliminó la app y se volvió a desplegar con Python 3.12.
