"""
Simulación básica de solicitudes para observar el monitoreo de la API.

Este script permite generar tráfico controlado hacia la API predictiva
de churn desarrollada durante el laboratorio.

Objetivos:
1. Enviar solicitudes válidas al endpoint POST /predict.
2. Enviar una solicitud atípica para generar alertas de datos.
3. Enviar una solicitud inválida para comprobar el error HTTP 422.
4. Mostrar la latencia informada por el middleware de la API.
5. Consultar el resumen acumulado mediante el endpoint GET /metrics.

Importante:
- La API debe estar activa antes de ejecutar este archivo.
- Este script no entrena el modelo.
- Este script no modifica la API.
- Solamente simula solicitudes para observar su comportamiento.
"""

# ============================================================
# BLOQUE 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================

from pprint import pprint
# pprint permite mostrar diccionarios JSON de forma ordenada y legible.

import requests
# requests permite enviar solicitudes HTTP desde Python.
# En este laboratorio se utiliza para comunicarse con la API local.

# ============================================================
# BLOQUE 2. CONFIGURACIÓN GENERAL
# ============================================================

# Dirección base donde se encuentra ejecutándose la API.
# El puerto predeterminado utilizado por Uvicorn es 8000.
BASE_URL = "http://127.0.0.1:8000"

# Tiempo máximo de espera para cada solicitud, expresado en segundos.
# Evita que el programa quede esperando indefinidamente si la API
# no responde o si existe algún problema de conexión.
TIMEOUT = 10

# ============================================================
# BLOQUE 3. CASOS DE PRUEBA
# ============================================================
# AQUÍ SE DEFINEN LAS SOLICITUDES QUE SE ENVIARÁN A POST /predict.
#
# Cada caso contiene:
# - nombre: etiqueta descriptiva para identificar la prueba;
# - datos: valores que recibirá la API.
#
# Se incluyen:
# - tres casos válidos dentro de rangos razonables;
# - un caso atípico aceptado técnicamente, pero fuera del histórico;
# - un caso inválido que debe ser rechazado con código HTTP 422.

CASOS = [
    {
        "nombre": "cliente_estable",
        "datos": {
            "antiguedad": 48,
            "cargo_mensual": 55.0,
            "reclamos": 0,
        },
    },
    {
        "nombre": "cliente_riesgo_medio",
        "datos": {
            "antiguedad": 18,
            "cargo_mensual": 110.0,
            "reclamos": 3,
        },
    },
    {
        "nombre": "cliente_alto_riesgo",
        "datos": {
            "antiguedad": 4,
            "cargo_mensual": 145.0,
            "reclamos": 7,
        },
    },
    {
        "nombre": "cliente_atipico",
        "datos": {
            "antiguedad": 180,
            "cargo_mensual": 600.0,
            "reclamos": 35,
        },
        # Este caso es técnicamente válido porque los valores respetan
        # los límites generales definidos por Pydantic.
        #
        # Sin embargo, los valores se encuentran fuera de los rangos
        # históricos del entrenamiento. Por ello, la API debe generar
        # alertas_datos.
    },
    {
        "nombre": "cliente_invalido",
        "datos": {
            "antiguedad": 12,
            "cargo_mensual": -50.0,
            "reclamos": 1,
        },
        # Este caso debe ser rechazado porque cargo_mensual es negativo.
        # La API devolverá un código HTTP 422 y aumentará el contador
        # errores_validacion.
    },
]

# ============================================================
# BLOQUE 4. FUNCIÓN PARA MOSTRAR LA RESPUESTA DE CADA CASO
# ============================================================
# AQUÍ SE MUESTRA:
# - nombre del caso;
# - código HTTP recibido;
# - latencia medida por el middleware;
# - contenido JSON devuelto por la API.

def mostrar_respuesta(
    nombre: str,
    respuesta: requests.Response,
) -> None:
    """
    Presenta de forma ordenada el resultado de una solicitud.

    Parámetros:
        nombre:
            Nombre descriptivo del caso evaluado.

        respuesta:
            Objeto Response devuelto por la librería requests.
            Contiene el código HTTP, las cabeceras y el cuerpo JSON.
    """

    print("\n" + "=" * 70)
    print(f"Caso: {nombre}")

    # Mostrar el código HTTP devuelto por la API.
    #
    # Ejemplos:
    # - 200: solicitud procesada correctamente;
    # - 422: error de validación de datos;
    # - 500: error interno de la API.
    print(f"Estado HTTP: {respuesta.status_code}")

    # Recuperar la cabecera agregada por el middleware de la API.
    #
    # En api/main.py se incorporó:
    # response.headers["X-Process-Time-ms"] = ...
    #
    # Esta cabecera informa cuántos milisegundos tardó la solicitud.
    latencia = respuesta.headers.get("X-Process-Time-ms")

    if latencia is not None:
        print(f"Latencia informada por API: {latencia} ms")
    else:
        print("Latencia informada por API: no disponible")

    # Intentar convertir el cuerpo de la respuesta a JSON.
    #
    # pprint permite mostrar la respuesta de forma ordenada.
    # Si la respuesta no contiene JSON válido, se muestra el texto original.
    try:
        pprint(respuesta.json())

    except requests.exceptions.JSONDecodeError:
        print("La respuesta no contiene un JSON válido.")
        print(respuesta.text)

# ============================================================
# BLOQUE 5. FUNCIÓN PARA ENVIAR UNA SOLICITUD A POST /predict
# ============================================================
# AQUÍ SE IMPLEMENTA:
# - envío de cada caso de prueba;
# - comunicación con POST /predict;
# - manejo de problemas de conexión.

def enviar_caso(caso: dict) -> None:
    """
    Envía un caso de prueba al endpoint POST /predict.

    Parámetro:
        caso:
            Diccionario con un nombre descriptivo y los datos del cliente.
    """

    nombre = caso["nombre"]
    datos = caso["datos"]

    try:
        # Enviar una solicitud HTTP POST.
        #
        # URL utilizada:
        # http://127.0.0.1:8000/predict
        #
        # El argumento json=datos convierte automáticamente el diccionario
        # de Python en el formato JSON esperado por la API.
        respuesta = requests.post(
            f"{BASE_URL}/predict",
            json=datos,
            timeout=TIMEOUT,
        )

        # Mostrar el resultado recibido.
        mostrar_respuesta(nombre, respuesta)

    except requests.exceptions.ConnectionError:
        # Este error aparece normalmente cuando la API no está activa.
        print("\n" + "=" * 70)
        print(f"Caso: {nombre}")
        print("Error: no fue posible conectarse con la API.")
        print("Verifique que Uvicorn se encuentre activo en otra terminal.")

    except requests.exceptions.Timeout:
        # Este error aparece si la API demora más del tiempo configurado.
        print("\n" + "=" * 70)
        print(f"Caso: {nombre}")
        print(f"Error: la API no respondió en menos de {TIMEOUT} segundos.")

    except requests.exceptions.RequestException as exc:
        # Captura otros problemas relacionados con la solicitud HTTP.
        print("\n" + "=" * 70)
        print(f"Caso: {nombre}")
        print(f"Error inesperado durante la solicitud: {exc}")

# ============================================================
# BLOQUE 6. FUNCIÓN PARA CONSULTAR GET /metrics
# ============================================================
# AQUÍ SE IMPLEMENTA:
# - consulta del resumen acumulado de métricas;
# - comunicación con el endpoint GET /metrics.

def consultar_metricas() -> None:
    """
    Consulta y muestra las métricas acumuladas por la API.

    El endpoint GET /metrics resume:
    - cantidad total de solicitudes;
    - errores de validación;
    - errores internos;
    - predicciones válidas;
    - resultados de alto y bajo riesgo;
    - solicitudes con anomalías;
    - latencia promedio y máxima;
    - distribución de códigos HTTP.
    """

    print("\n" + "=" * 70)
    print("Resumen acumulado de métricas")

    try:
        # Enviar una solicitud HTTP GET al endpoint /metrics.
        respuesta_metricas = requests.get(
            f"{BASE_URL}/metrics",
            timeout=TIMEOUT,
        )

        print(f"Estado HTTP: {respuesta_metricas.status_code}")

        # Mostrar el resumen JSON de manera ordenada.
        pprint(respuesta_metricas.json())

    except requests.exceptions.ConnectionError:
        print("Error: no fue posible consultar las métricas.")
        print("Verifique que la API se encuentre activa.")

    except requests.exceptions.Timeout:
        print(
            f"Error: la API no respondió en menos de {TIMEOUT} segundos."
        )

    except requests.exceptions.JSONDecodeError:
        print("Error: la respuesta de /metrics no contiene un JSON válido.")

    except requests.exceptions.RequestException as exc:
        print(f"Error inesperado durante la consulta: {exc}")

# ============================================================
# BLOQUE 7. FUNCIÓN PRINCIPAL
# ============================================================
# AQUÍ SE DEFINE EL ORDEN COMPLETO DE EJECUCIÓN:
# 1. Mostrar un encabezado.
# 2. Recorrer todos los casos de prueba.
# 3. Enviar cada solicitud a POST /predict.
# 4. Consultar GET /metrics al finalizar.

def main() -> None:
    """
    Ejecuta la simulación completa de solicitudes.
    """

    print("=" * 70)
    print("SIMULACIÓN DE SOLICITUDES PARA LA API PREDICTIVA")
    print("=" * 70)

    # Recorrer secuencialmente todos los casos definidos anteriormente.
    for caso in CASOS:
        enviar_caso(caso)

    # Consultar el resumen acumulado después de procesar las solicitudes.
    consultar_metricas()

# ============================================================
# BLOQUE 8. PUNTO DE ENTRADA DEL PROGRAMA
# ============================================================
# Esta condición permite ejecutar main() únicamente cuando este archivo
# se inicia directamente desde PowerShell.
#
# Comando:
# python tests\simular_solicitudes.py
#
# Si el archivo fuera importado desde otro script, main() no se ejecutaría
# automáticamente.

if __name__ == "__main__":
    main()
