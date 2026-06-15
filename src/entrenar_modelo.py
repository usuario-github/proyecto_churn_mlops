"""
Entrenamiento de un modelo académico de predicción de churn.

El script genera datos sintéticos reproducibles, entrena un modelo
de clasificación y guarda el artefacto serializado con joblib.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Rutas robustas basadas en la ubicación del archivo actual
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

MODEL_PATH = MODELS_DIR / "modelo_churn_v1.joblib"
METADATA_PATH = MODELS_DIR / "modelo_churn_v1_metadata.json"
METRICS_PATH = DOCS_DIR / "metricas_modelo.md"

def generar_datos_sinteticos(n_registros: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera datos académicos para simular abandono de clientes.

    Variables:
    - antiguedad: meses como cliente
    - cargo_mensual: monto mensual pagado
    - reclamos: cantidad de reclamos recientes
    """
    rng = np.random.default_rng(seed=42)

    antiguedad = rng.integers(1, 73, size=n_registros)
    cargo_mensual = rng.uniform(20, 150, size=n_registros)
    reclamos = rng.integers(0, 8, size=n_registros)

    # Regla sintética: mayor cargo y más reclamos elevan el riesgo;
    # mayor antigüedad reduce el riesgo.
    puntaje_riesgo = (
        -0.045 * antiguedad
        + 0.025 * cargo_mensual
        + 0.65 * reclamos
        - 1.8
    )

    probabilidad = 1 / (1 + np.exp(-puntaje_riesgo))
    churn = rng.binomial(1, probabilidad)

    X = np.column_stack([antiguedad, cargo_mensual, reclamos])
    y = churn

    return X, y

def entrenar_y_guardar_modelo() -> None:
    """Entrena, evalúa y guarda el modelo junto con sus metadatos."""
    MODELS_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    X, y = generar_datos_sinteticos()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    modelo = Pipeline(
        steps=[
            ("escalado", StandardScaler()),
            ("clasificador", LogisticRegression(random_state=42)),
        ]
    )

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "accuracy": round(float(accuracy_score(y_test, predicciones)), 4),
        "f1_score": round(float(f1_score(y_test, predicciones)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, probabilidades)), 4),
    }

    joblib.dump(modelo, MODEL_PATH)

    metadata = {
        "version_modelo": "modelo_churn_v1",
        "archivo_modelo": MODEL_PATH.name,
        "variables_entrada": [
            "antiguedad",
            "cargo_mensual",
            "reclamos",
        ],
        "version_sklearn": sklearn.__version__,
        "metricas": metricas,
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    METRICS_PATH.write_text(
        "# Métricas del modelo\n\n"
        f"- Modelo: `{MODEL_PATH.name}`\n"
        f"- Accuracy: {metricas['accuracy']}\n"
        f"- F1-score: {metricas['f1_score']}\n"
        f"- AUC-ROC: {metricas['auc_roc']}\n"
        f"- Versión scikit-learn: {sklearn.__version__}\n",
        encoding="utf-8",
    )

    print("Modelo generado correctamente.")
    print(f"Ruta del modelo: {MODEL_PATH}")
    print(f"Métricas: {metricas}")

if __name__ == "__main__":
    entrenar_y_guardar_modelo()
