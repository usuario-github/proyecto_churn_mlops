from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Datos de prueba reutilizables
# ---------------------------------------------------------------------------

CLIENTE_BAJO_RIESGO = {
    "antiguedad": 48,
    "cargo_mensual": 55.0,
    "reclamos": 0,
}

CLIENTE_ALTO_RIESGO = {
    "antiguedad": 4,
    "cargo_mensual": 145.0,
    "reclamos": 7,
}

CLIENTE_NUEVO_SIN_RECLAMOS = {
    "antiguedad": 0,
    "cargo_mensual": 80.0,
    "reclamos": 0,
}

# ---------------------------------------------------------------------------
# Tests existentes: GET / y GET /health
# ---------------------------------------------------------------------------

def test_inicio():
    response = client.get("/")

    assert response.status_code == 200
    assert "mensaje" in response.json()

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert "estado" in response.json()
    assert "modelo" in response.json()

# ---------------------------------------------------------------------------
# POST /predict — estructura de la respuesta
# ---------------------------------------------------------------------------

def test_predict_devuelve_campos_requeridos():
    """La respuesta debe contener todos los campos definidos en PrediccionSalida."""
    response = client.post("/predict", json=CLIENTE_BAJO_RIESGO)

    assert response.status_code == 200
    data = response.json()
    assert "prediccion" in data
    assert "probabilidad" in data
    assert "version_modelo" in data
    assert "autor" in data
    assert "alertas_datos" in data

def test_predict_probabilidad_entre_0_y_1():
    """La probabilidad de churn siempre debe estar en el rango [0, 1]."""
    response = client.post("/predict", json=CLIENTE_BAJO_RIESGO)

    assert response.status_code == 200
    probabilidad = response.json()["probabilidad"]
    assert 0.0 <= probabilidad <= 1.0

def test_predict_etiqueta_es_valor_esperado():
    """La predicción solo puede ser 'alto_riesgo' o 'bajo_riesgo'."""
    response = client.post("/predict", json=CLIENTE_BAJO_RIESGO)

    assert response.status_code == 200
    assert response.json()["prediccion"] in {"alto_riesgo", "bajo_riesgo"}

# ---------------------------------------------------------------------------
# POST /predict — lógica de predicción
# ---------------------------------------------------------------------------

def test_predict_cliente_estable_es_bajo_riesgo():
    """Cliente con alta antigüedad y cero reclamos debe clasificarse como bajo riesgo."""
    response = client.post("/predict", json=CLIENTE_BAJO_RIESGO)

    assert response.status_code == 200
    assert response.json()["prediccion"] == "bajo_riesgo"

def test_predict_cliente_critico_es_alto_riesgo():
    """Cliente nuevo con cargo alto y muchos reclamos debe clasificarse como alto riesgo."""
    response = client.post("/predict", json=CLIENTE_ALTO_RIESGO)

    assert response.status_code == 200
    assert response.json()["prediccion"] == "alto_riesgo"

# ---------------------------------------------------------------------------
# POST /predict — detección de anomalías
# ---------------------------------------------------------------------------

def test_predict_cliente_atipico_genera_alertas():
    """Valores fuera del rango histórico deben generar alertas sin rechazar la solicitud."""
    cliente_atipico = {
        "antiguedad": 180,
        "cargo_mensual": 600.0,
        "reclamos": 35,
    }
    response = client.post("/predict", json=cliente_atipico)

    assert response.status_code == 200
    assert len(response.json()["alertas_datos"]) > 0

def test_predict_cliente_normal_sin_alertas():
    """Valores dentro del rango histórico no deben generar alertas."""
    response = client.post("/predict", json=CLIENTE_BAJO_RIESGO)

    assert response.status_code == 200
    assert response.json()["alertas_datos"] == []

# ---------------------------------------------------------------------------
# POST /predict — validación de campos individuales
# ---------------------------------------------------------------------------

def test_predict_cargo_negativo_retorna_422():
    """Un cargo mensual negativo viola las reglas de campo y debe ser rechazado."""
    cliente_invalido = {
        "antiguedad": 12,
        "cargo_mensual": -50.0,
        "reclamos": 1,
    }
    response = client.post("/predict", json=cliente_invalido)

    assert response.status_code == 422

def test_predict_reclamos_negativos_retorna_422():
    """Reclamos negativos no son un valor posible y deben ser rechazados."""
    cliente_invalido = {
        "antiguedad": 12,
        "cargo_mensual": 80.0,
        "reclamos": -1,
    }
    response = client.post("/predict", json=cliente_invalido)

    assert response.status_code == 422

def test_predict_campo_faltante_retorna_422():
    """Omitir un campo requerido debe producir un error de validación."""
    cliente_incompleto = {
        "antiguedad": 12,
        "cargo_mensual": 80.0,
    }
    response = client.post("/predict", json=cliente_incompleto)

    assert response.status_code == 422

# ---------------------------------------------------------------------------
# POST /predict — validación cruzada entre campos
# ---------------------------------------------------------------------------

def test_predict_cliente_nuevo_con_reclamos_retorna_422():
    """Un cliente nuevo (antiguedad=0) no puede tener reclamos previos."""
    cliente_incoherente = {
        "antiguedad": 0,
        "cargo_mensual": 80.0,
        "reclamos": 3,
    }
    response = client.post("/predict", json=cliente_incoherente)

    assert response.status_code == 422

def test_predict_cliente_nuevo_sin_reclamos_es_valido():
    """Un cliente nuevo con cero reclamos debe ser aceptado."""
    response = client.post("/predict", json=CLIENTE_NUEVO_SIN_RECLAMOS)

    assert response.status_code == 200
