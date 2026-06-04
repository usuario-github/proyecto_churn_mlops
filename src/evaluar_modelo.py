from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

TEST_DATA = DATA_DIR / "test.csv"
MODEL_FILE = MODELS_DIR / "modelo_churn.pkl"
METRICS_FILE = DOCS_DIR / "metricas_modelo.md"

def evaluar_modelo():
    """
    Evalúa el modelo entrenado y guarda las métricas principales.
    """

    if not TEST_DATA.exists():
        raise FileNotFoundError(
            "No se encontró data/test.csv. Primero ejecuta src/preparar_datos.py"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "No se encontró el modelo entrenado. Primero ejecuta src/entrenar_modelo.py"
        )

    DOCS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(TEST_DATA)

    X_test = df.drop(columns=["churn"])
    y_test = df["churn"]

    modelo = joblib.load(MODEL_FILE)

    y_pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    contenido = f"""# Métricas del modelo de churn

## Resultados principales

| Métrica | Valor |
|---|---:|
| Accuracy | {accuracy:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1-score | {f1:.4f} |

## Interpretación inicial

Estas métricas permiten evaluar el desempeño inicial del modelo de clasificación.

- Accuracy indica el porcentaje general de aciertos.
- Precision indica qué tan confiables son las predicciones positivas.
- Recall indica qué proporción de clientes con churn fueron identificados.
- F1-score resume precision y recall en una sola métrica.
"""

    METRICS_FILE.write_text(contenido, encoding="utf-8")

    print("Modelo evaluado correctamente.")
    print(f"Métricas guardadas en: {METRICS_FILE}")

if __name__ == "__main__":
    evaluar_modelo()
