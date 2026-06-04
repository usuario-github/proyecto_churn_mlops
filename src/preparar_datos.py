from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

RAW_DATA = DATA_DIR / "churn_clientes.csv"
TRAIN_DATA = DATA_DIR / "train.csv"
TEST_DATA = DATA_DIR / "test.csv"

def crear_dataset_demo():
    """
    Crea un dataset pequeño de ejemplo para la práctica.
    Este dataset permite ejecutar el flujo inicial sin descargar datos externos.
    """

    datos = [
        {"edad": 25, "antiguedad_meses": 6, "saldo_promedio": 1200, "reclamos": 3, "usa_app": 0, "churn": 1},
        {"edad": 34, "antiguedad_meses": 24, "saldo_promedio": 3500, "reclamos": 0, "usa_app": 1, "churn": 0},
        {"edad": 45, "antiguedad_meses": 36, "saldo_promedio": 5000, "reclamos": 1, "usa_app": 1, "churn": 0},
        {"edad": 22, "antiguedad_meses": 4, "saldo_promedio": 800, "reclamos": 4, "usa_app": 0, "churn": 1},
        {"edad": 52, "antiguedad_meses": 60, "saldo_promedio": 7000, "reclamos": 0, "usa_app": 1, "churn": 0},
        {"edad": 29, "antiguedad_meses": 8, "saldo_promedio": 1500, "reclamos": 2, "usa_app": 0, "churn": 1},
        {"edad": 40, "antiguedad_meses": 30, "saldo_promedio": 4200, "reclamos": 1, "usa_app": 1, "churn": 0},
        {"edad": 31, "antiguedad_meses": 10, "saldo_promedio": 1600, "reclamos": 3, "usa_app": 0, "churn": 1},
        {"edad": 48, "antiguedad_meses": 48, "saldo_promedio": 6000, "reclamos": 0, "usa_app": 1, "churn": 0},
        {"edad": 27, "antiguedad_meses": 7, "saldo_promedio": 1100, "reclamos": 4, "usa_app": 0, "churn": 1},
        {"edad": 36, "antiguedad_meses": 26, "saldo_promedio": 3900, "reclamos": 1, "usa_app": 1, "churn": 0},
        {"edad": 23, "antiguedad_meses": 5, "saldo_promedio": 900, "reclamos": 5, "usa_app": 0, "churn": 1},
        {"edad": 55, "antiguedad_meses": 72, "saldo_promedio": 8200, "reclamos": 0, "usa_app": 1, "churn": 0},
        {"edad": 33, "antiguedad_meses": 14, "saldo_promedio": 2100, "reclamos": 2, "usa_app": 0, "churn": 1},
        {"edad": 41, "antiguedad_meses": 33, "saldo_promedio": 4600, "reclamos": 0, "usa_app": 1, "churn": 0},
        {"edad": 30, "antiguedad_meses": 9, "saldo_promedio": 1300, "reclamos": 3, "usa_app": 0, "churn": 1},
    ]

    df = pd.DataFrame(datos)
    df.to_csv(RAW_DATA, index=False)

def preparar_datos():
    """
    Prepara los datos para entrenamiento y prueba.
    """

    DATA_DIR.mkdir(exist_ok=True)

    if not RAW_DATA.exists():
        crear_dataset_demo()

    df = pd.read_csv(RAW_DATA)

    df = df.drop_duplicates()
    df = df.dropna()

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["churn"]
    )

    train_df.to_csv(TRAIN_DATA, index=False)
    test_df.to_csv(TEST_DATA, index=False)

    print("Datos preparados correctamente.")
    print(f"Archivo de entrenamiento: {TRAIN_DATA}")
    print(f"Archivo de prueba: {TEST_DATA}")

if __name__ == "__main__":
    preparar_datos()
