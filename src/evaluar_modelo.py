from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

MODEL_FILE = MODELS_DIR / "modelo_churn_v1.joblib"
METRICS_FILE = DOCS_DIR / "metricas_modelo.md"

def _generar_datos_test() -> tuple[np.ndarray, np.ndarray]:
    """Reproduce el mismo conjunto de prueba generado en entrenar_modelo.py."""
    rng = np.random.default_rng(seed=42)
    n = 800
    antiguedad = rng.integers(1, 73, size=n)
    cargo_mensual = rng.uniform(20, 150, size=n)
    reclamos = rng.integers(0, 8, size=n)
    puntaje = -0.045 * antiguedad + 0.025 * cargo_mensual + 0.65 * reclamos - 1.8
    prob = 1 / (1 + np.exp(-puntaje))
    churn = rng.binomial(1, prob)
    X = np.column_stack([antiguedad, cargo_mensual, reclamos])
    _, X_test, _, y_test = train_test_split(
        X, churn, test_size=0.25, random_state=42, stratify=churn
    )
    return X_test, y_test

def evaluar_modelo():
    """
    Evalúa el modelo entrenado y guarda las métricas principales.
    """

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "No se encontró el modelo entrenado. Primero ejecuta src/entrenar_modelo.py"
        )

    DOCS_DIR.mkdir(exist_ok=True)

    X_test, y_test = _generar_datos_test()

    modelo = joblib.load(MODEL_FILE)

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    contenido = f"""# Métricas del modelo de churn

## Resultados principales

| Métrica | Valor |
|---|---:|
| Accuracy | {accuracy:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1-score | {f1:.4f} |
| ROC-AUC | {roc_auc:.4f} |

## Interpretación inicial

Estas métricas permiten evaluar el desempeño inicial del modelo de clasificación.

- Accuracy indica el porcentaje general de aciertos.
- Precision indica qué tan confiables son las predicciones positivas.
- Recall indica qué proporción de clientes con churn fueron identificados.
- F1-score resume precision y recall en una sola métrica.
- ROC-AUC mide la capacidad del modelo para separar las clases. Un valor de 0.5 equivale a azar; 1.0 es perfecto.
"""

    METRICS_FILE.write_text(contenido, encoding="utf-8")

    print("Modelo evaluado correctamente.")
    print(f"Métricas guardadas en: {METRICS_FILE}")

if __name__ == "__main__":
    evaluar_modelo()
