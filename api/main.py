"""
API predictiva de churn con monitoreo básico.

Esta versión incorpora una primera capa académica de observabilidad:

1. Logging a archivo y consola.
2. Medición de latencia mediante middleware.
3. Conteo acumulado de solicitudes, errores y predicciones.
4. Detección de valores fuera del rango histórico.
5. Endpoint GET /metrics para consultar un resumen acumulado.

Importante:
- Las métricas se almacenan temporalmente en memoria.
- Si la API se reinicia, los contadores vuelven a cero.
- Esta solución tiene fines académicos y no reemplaza una plataforma
  empresarial de monitoreo.
"""

# ============================================================
# BLOQUE 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================
# Cada librería cumple una función específica dentro de la API.

from collections import Counter
# Counter permite contar cuántas respuestas HTTP 200, 422, 500, etc. se generan.

from pathlib import Path
# Path permite construir rutas compatibles con Windows de forma segura.

from threading import Lock
# Lock evita inconsistencias cuando varias solicitudes intentan actualizar
# las métricas al mismo tiempo.

from time import perf_counter
# perf_counter permite medir con precisión cuánto tarda una solicitud.

import logging
# logging permite registrar eventos en consola y en un archivo.

import joblib
# joblib permite cargar el modelo serializado previamente.

from fastapi import FastAPI, HTTPException, Request
# FastAPI crea la aplicación.
# Request permite acceder a la información de cada solicitud HTTP.
# HTTPException permite devolver errores controlados.

from fastapi.exception_handlers import request_validation_exception_handler
# Permite conservar la respuesta estándar de FastAPI cuando ocurre
# un error de validación.

from fastapi.exceptions import RequestValidationError
# Representa errores como campos faltantes, valores negativos
# o tipos de datos incorrectos.

from pydantic import BaseModel, Field, model_validator
# BaseModel define la estructura esperada de los datos.
# Field agrega reglas de validación.

# ============================================================
# BLOQUE 2. CONFIGURACIÓN GENERAL DEL PROYECTO
# ============================================================

# Ruta raíz del proyecto.
# __file__ representa api/main.py.
# parents[1] permite subir desde api/ hasta proyecto_churn_mlops/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ruta del modelo serializado generado en sesiones anteriores.
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_churn_v1.joblib"

# Carpeta donde se almacenará el archivo de logs.
LOGS_DIR = PROJECT_ROOT / "logs"

# Ruta completa del archivo que registrará los eventos.
LOG_FILE = LOGS_DIR / "monitor_api.log"

# Información que se mostrará en las respuestas de la API.
VERSION_MODELO = "modelo_churn_v1"

# Personalizar obligatoriamente con nombre y apellido.
AUTOR = "Rodny Hurtado"

# ============================================================
# BLOQUE 3. RANGOS HISTÓRICOS DE REFERENCIA
# ============================================================
# Estos límites representan los valores observados durante la generación
# de los datos de entrenamiento.
#
# La API admite valores más amplios, pero genera una alerta cuando una entrada
# se aparta del comportamiento histórico esperado.
#
# Esta verificación constituye una señal inicial de posible drift.
# No equivale todavía a una prueba estadística formal.

RANGOS_HISTORICOS = {
    "antiguedad": (1, 72),
    "cargo_mensual": (20.0, 150.0),
    "reclamos": (0, 7),
}

# ============================================================
# BLOQUE 4. LOGGING A ARCHIVO Y CONSOLA
# ============================================================
# AQUÍ SE IMPLEMENTA:
# - logging a archivo;
# - logging en consola.
#
# La carpeta logs/ se crea automáticamente si todavía no existe.

LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    # INFO permite registrar eventos normales, advertencias y errores.

    format="%(asctime)s | %(levelname)s | %(message)s",
    # Cada registro mostrará:
    # fecha y hora | nivel del evento | mensaje

    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        # Registra los eventos en:
        # logs/monitor_api.log

        logging.StreamHandler(),
        # Muestra los mismos eventos en la terminal.
    ],
)

# Logger específico para esta aplicación.
logger = logging.getLogger("api_churn")

# ============================================================
# BLOQUE 5. VERIFICACIÓN Y CARGA DEL MODELO
# ============================================================

# Antes de iniciar la API, comprobar que el modelo existe.
if not MODEL_PATH.exists():
    raise RuntimeError(
        "No se encontró el modelo serializado. "
        "Ejecute primero: python src\\entrenar_modelo.py"
    )

# Cargar el modelo entrenado desde el archivo .joblib.
modelo = joblib.load(MODEL_PATH)

# Registrar en consola y archivo que el modelo fue cargado.
logger.info("Modelo cargado correctamente: %s", VERSION_MODELO)

# ============================================================
# BLOQUE 6. CONTADORES DE MÉTRICAS EN MEMORIA
# ============================================================
# AQUÍ SE PREPARA EL CONTEO DE:
# - solicitudes HTTP;
# - errores de validación;
# - errores internos;
# - predicciones válidas;
# - predicciones de alto y bajo riesgo;
# - solicitudes con anomalías;
# - latencia promedio y máxima;
# - códigos HTTP.
#
# Los valores comienzan en cero cuando se inicia la API.

metricas = {
    "solicitudes_totales": 0,
    "errores_validacion": 0,
    "errores_internos": 0,
    "predicciones_validas": 0,
    "predicciones_alto_riesgo": 0,
    "predicciones_bajo_riesgo": 0,
    "solicitudes_con_anomalias": 0,
    "latencia_acumulada_ms": 0.0,
    "latencia_maxima_ms": 0.0,
    "codigos_http": Counter(),
}

# Lock evita que dos solicitudes modifiquen simultáneamente
# los mismos contadores.
metricas_lock = Lock()

# ============================================================
# BLOQUE 7. MODELOS DE DATOS Y VALIDACIÓN DE ENTRADAS
# ============================================================

class ClienteEntrada(BaseModel):
    """
    Define los datos requeridos por POST /predict.

    Los límites siguientes son reglas generales de validación técnica.
    Son más amplios que los rangos históricos del entrenamiento.

    Ejemplo:
    - antiguedad=180 es técnicamente válida porque es menor que 240;
    - sin embargo, generará una alerta porque supera el máximo histórico de 72.
    """

    antiguedad: int = Field(..., ge=0, le=240)
    cargo_mensual: float = Field(..., ge=0, le=5000)
    reclamos: int = Field(..., ge=0, le=100)

    @model_validator(mode="after")
    def validar_coherencia(self) -> "ClienteEntrada":
        if self.antiguedad == 0 and self.reclamos > 0:
            raise ValueError(
                "Un cliente nuevo (antiguedad=0) no puede tener reclamos previos."
            )
        return self

class PrediccionSalida(BaseModel):
    """
    Define la estructura de respuesta de POST /predict.
    """

    prediccion: str
    probabilidad: float
    version_modelo: str
    autor: str
    alertas_datos: list[str]

# ============================================================
# BLOQUE 8. DETECCIÓN DE VALORES FUERA DEL RANGO HISTÓRICO
# ============================================================
# AQUÍ SE IMPLEMENTA:
# - detección de valores fuera del rango histórico.
#
# La función compara cada valor recibido con los valores mínimos y máximos
# definidos en RANGOS_HISTORICOS.

def detectar_anomalias(datos: ClienteEntrada) -> list[str]:
    """
    Identifica valores técnicamente permitidos por la API, pero atípicos
    frente al histórico utilizado durante el entrenamiento.

    Devuelve:
        Una lista de mensajes de alerta.

    Ejemplo:
        Si cargo_mensual=600, la API acepta el dato porque es menor que 5000,
        pero genera una alerta porque supera el rango histórico [20.0, 150.0].
    """

    alertas: list[str] = []

    # Convertir el objeto recibido en un diccionario.
    valores = datos.model_dump()

    # Revisar individualmente cada variable.
    for variable, valor in valores.items():
        minimo, maximo = RANGOS_HISTORICOS[variable]

        # Comprobar si el valor está fuera del rango histórico.
        if valor < minimo or valor > maximo:
            alertas.append(
                f"{variable}={valor} fuera del rango histórico "
                f"[{minimo}, {maximo}]"
            )

    return alertas

# ============================================================
# BLOQUE 9. PREPARACIÓN DEL RESUMEN DE MÉTRICAS
# ============================================================
# AQUÍ SE PREPARA LA INFORMACIÓN QUE DEVOLVERÁ GET /metrics.
#
# La latencia promedio se calcula dividiendo:
# latencia acumulada / número total de solicitudes procesadas.

def resumen_metricas() -> dict:
    """
    Devuelve una copia segura y legible de las métricas acumuladas.

    La función utiliza metricas_lock para evitar que los valores sean
    modificados mientras se construye la respuesta.
    """

    with metricas_lock:
        total = metricas["solicitudes_totales"]

        # Evitar división entre cero cuando todavía no existen solicitudes.
        latencia_promedio = (
            metricas["latencia_acumulada_ms"] / total
            if total
            else 0.0
        )

        return {
            "version_modelo": VERSION_MODELO,
            "autor": AUTOR,
            "solicitudes_totales": total,
            "errores_validacion": metricas["errores_validacion"],
            "errores_internos": metricas["errores_internos"],
            "predicciones_validas": metricas["predicciones_validas"],
            "predicciones_alto_riesgo": metricas[
                "predicciones_alto_riesgo"
            ],
            "predicciones_bajo_riesgo": metricas[
                "predicciones_bajo_riesgo"
            ],
            "solicitudes_con_anomalias": metricas[
                "solicitudes_con_anomalias"
            ],
            "latencia_promedio_ms": round(latencia_promedio, 3),
            "latencia_maxima_ms": round(
                metricas["latencia_maxima_ms"], 3
            ),
            "codigos_http": dict(metricas["codigos_http"]),
        }

# ============================================================
# BLOQUE 10. CREACIÓN DE LA APLICACIÓN FASTAPI
# ============================================================

app = FastAPI(
    title="API de predicción de churn con monitoreo básico",
    description="Servicio académico ML-Ops con métricas y logs.",
    version="2.0.0",
)

# ============================================================
# BLOQUE 11. MIDDLEWARE PARA MEDIR LATENCIA Y CONTAR SOLICITUDES
# ============================================================
# AQUÍ SE IMPLEMENTA:
# - medición de latencia mediante middleware;
# - conteo de solicitudes;
# - conteo de códigos HTTP;
# - detección de errores internos no controlados.
#
# El middleware se ejecuta automáticamente cada vez que la API recibe
# una solicitud HTTP: /, /health, /metrics, /docs o /predict.

@app.middleware("http")
async def registrar_solicitud(request: Request, call_next):
    """
    Observa todas las solicitudes HTTP procesadas por la API.

    Procedimiento:
    1. Registrar el instante inicial.
    2. Permitir que FastAPI procese la solicitud.
    3. Calcular el tiempo transcurrido.
    4. Actualizar los contadores.
    5. Registrar el evento en consola y archivo.
    6. Agregar la latencia como cabecera HTTP.
    """

    # Guardar el instante exacto en que comienza la solicitud.
    inicio = perf_counter()

    try:
        # Continuar con el procesamiento normal de la solicitud.
        response = await call_next(request)

    except Exception:
        # Si ocurre un error inesperado, incrementar el contador.
        with metricas_lock:
            metricas["errores_internos"] += 1

        # Registrar el error completo en consola y archivo.
        logger.exception(
            "Error interno no controlado en %s",
            request.url.path,
        )

        # Permitir que FastAPI continúe gestionando el error.
        raise

    # Calcular la latencia en milisegundos.
    latencia_ms = (perf_counter() - inicio) * 1000

    # Actualizar métricas acumuladas.
    with metricas_lock:
        metricas["solicitudes_totales"] += 1
        metricas["latencia_acumulada_ms"] += latencia_ms

        metricas["latencia_maxima_ms"] = max(
            metricas["latencia_maxima_ms"],
            latencia_ms,
        )

        # Registrar la cantidad de respuestas por código HTTP.
        # Ejemplos: 200, 422 y 500.
        metricas["codigos_http"][str(response.status_code)] += 1

    # Registrar información de la solicitud en consola y archivo.
    logger.info(
        "Solicitud | metodo=%s | ruta=%s | estado=%s | latencia_ms=%.3f",
        request.method,
        request.url.path,
        response.status_code,
        latencia_ms,
    )

    # Agregar la latencia a las cabeceras HTTP de la respuesta.
    # Esto permite verla desde Swagger o desde el script de simulación.
    response.headers["X-Process-Time-ms"] = f"{latencia_ms:.3f}"

    return response

# ============================================================
# BLOQUE 12. MANEJO DE ERRORES DE VALIDACIÓN
# ============================================================
# AQUÍ SE IMPLEMENTA:
# - conteo de errores producidos por datos faltantes;
# - conteo de valores no permitidos;
# - registro del error en consola y archivo.
#
# Ejemplos:
# - falta el campo reclamos;
# - cargo_mensual contiene un número negativo;
# - antiguedad contiene texto en lugar de un número.

@app.exception_handler(RequestValidationError)
async def registrar_error_validacion(
    request: Request,
    exc: RequestValidationError,
):
    """
    Incrementa el contador cuando FastAPI rechaza los datos de entrada.
    """

    with metricas_lock:
        metricas["errores_validacion"] += 1

    logger.warning(
        "Error de validación | ruta=%s | detalle=%s",
        request.url.path,
        exc.errors(),
    )

    # Mantener la respuesta estándar HTTP 422 de FastAPI.
    return await request_validation_exception_handler(request, exc)

# ============================================================
# BLOQUE 13. ENDPOINT DE INICIO
# ============================================================

@app.get("/")
def inicio() -> dict[str, str]:
    """
    Confirma que el servicio está activo.
    """

    return {
        "mensaje": "Servicio ML-Ops activo",
        "estado": "ok",
        "autor": AUTOR,
    }

# ============================================================
# BLOQUE 14. ENDPOINT DE SALUD
# ============================================================

@app.get("/health")
def health() -> dict[str, str]:
    """
    Confirma que la API funciona y que el monitoreo está activo.
    """

    return {
        "estado": "ok",
        "modelo": VERSION_MODELO,
        "monitoreo": "activo",
    }

# ============================================================
# BLOQUE 15. ENDPOINT GET /metrics
# ============================================================
# AQUÍ SE IMPLEMENTA:
# - endpoint GET /metrics para consultar el resumen acumulado.
#
# Puede abrirse desde:
# http://127.0.0.1:8000/metrics

@app.get("/metrics")
def metrics() -> dict:
    """
    Devuelve las métricas acumuladas desde que se inició la API.
    """

    return resumen_metricas()

# ============================================================
# BLOQUE 16. ENDPOINT POST /predict
# ============================================================
# AQUÍ SE IMPLEMENTA:
# - predicción del modelo;
# - conteo de predicciones válidas;
# - conteo de predicciones de alto y bajo riesgo;
# - conteo de solicitudes con anomalías;
# - registro de predicciones y alertas en consola y archivo.

@app.post("/predict", response_model=PrediccionSalida)
def predict(datos: ClienteEntrada) -> PrediccionSalida:
    """
    Recibe los datos del cliente y genera una predicción de churn.

    Flujo:
    1. Detectar valores fuera del rango histórico.
    2. Construir la entrada esperada por el modelo.
    3. Calcular la probabilidad.
    4. Asignar alto_riesgo o bajo_riesgo.
    5. Actualizar métricas.
    6. Registrar eventos.
    7. Devolver la respuesta.
    """

    try:
        # Paso 1. Detectar datos atípicos.
        alertas = detectar_anomalias(datos)

        # Paso 2. Preparar los datos con el orden utilizado al entrenar:
        # antiguedad, cargo_mensual y reclamos.
        X = [[
            datos.antiguedad,
            datos.cargo_mensual,
            datos.reclamos,
        ]]

        # Paso 3. Calcular la probabilidad de abandono.
        probabilidad = float(modelo.predict_proba(X)[0][1])

        # Paso 4. Aplicar un umbral de decisión del 50 %.
        etiqueta = (
            "alto_riesgo"
            if probabilidad >= 0.50
            else "bajo_riesgo"
        )

        # Paso 5. Actualizar las métricas de predicción.
        with metricas_lock:
            metricas["predicciones_validas"] += 1
            metricas[f"predicciones_{etiqueta}"] += 1

            if alertas:
                metricas["solicitudes_con_anomalias"] += 1

        # Paso 6. Registrar una advertencia si existen datos atípicos.
        if alertas:
            logger.warning(
                "Valores fuera de rango histórico: %s",
                alertas,
            )

        # Registrar la predicción en consola y archivo.
        logger.info(
            "Predicción | resultado=%s | probabilidad=%.4f | alertas=%s",
            etiqueta,
            probabilidad,
            len(alertas),
        )

        # Paso 7. Devolver la respuesta.
        return PrediccionSalida(
            prediccion=etiqueta,
            probabilidad=round(probabilidad, 4),
            version_modelo=VERSION_MODELO,
            autor=AUTOR,
            alertas_datos=alertas,
        )

    except Exception as exc:
        # Incrementar el contador si ocurre un error durante la inferencia.
        with metricas_lock:
            metricas["errores_internos"] += 1

        # Registrar el detalle técnico en consola y archivo.
        logger.exception("No fue posible generar la predicción")

        # Devolver una respuesta controlada al cliente.
        raise HTTPException(
            status_code=500,
            detail="No fue posible generar la predicción.",
        ) from exc
