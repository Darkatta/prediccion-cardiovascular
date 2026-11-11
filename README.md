# Predicción de enfermedad cardiovascular — Despliegue en Streamlit

Proyecto de aula de **Minería de Datos** (Universidad de la Costa, CUC) — Tercer corte.

**Aplicación en línea:** [PEGAR AQUÍ LA URL DE STREAMLIT]

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
| Caso normal | [completar: valores ingresados y resultado] |
| Entrada inválida (ej. edad = 250) | [completar: mensaje que mostró la app] |
| Registro conocido del conjunto de prueba (índice [n]) | Notebook: P = [__] · Streamlit: P = [__] → coinciden |

## Limitaciones
Dataset pequeño (303) y prueba de 61 pacientes; sin optimización de hiperparámetros; `ca` y `thal`
vienen de pruebas diagnósticas adicionales; persisten falsos negativos; datos de una sola institución.

## Uso de inteligencia artificial
**Herramienta:** Claude (Anthropic)
**Propósito:** [completar: p. ej., generar la estructura de app.py, la validación de entradas y el guardado del pipeline]
**Prompt principal:** [pegar el prompt real]
**Cómo se verificó la respuesta:** [completar: p. ej., ejecución de la celda en el notebook, comparación de la predicción de Streamlit con la del notebook para el mismo registro, prueba de entradas inválidas]
**Problemas encontrados:** [completar con lo que realmente pasó]
**Ajustes realizados:** [completar]
