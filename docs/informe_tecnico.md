# Informe Técnico — Proyecto MLOps: Predicción de Churn

**Autor:** Rodny Hurtado
**Curso:** ML-Ops y Puesta en Producción — Semana 2
**Fecha:** 17 de junio de 2026
**Versión del modelo:** modelo_churn_v1

---

## 1. Problema y objetivo

### Problema

Las empresas de servicios pierden ingresos cuando los clientes abandonan la plataforma sin previo aviso. Detectar este comportamiento de manera reactiva —después de que ocurre— impide tomar acciones de retención a tiempo.

### Objetivo

Construir un sistema de predicción de abandono de clientes (*churn*) que cumpla con las prácticas fundamentales de MLOps:

- Entrenar un modelo de clasificación reproducible.
- Exponer el modelo como un servicio HTTP con validación de entradas.
- Incorporar observabilidad básica: logging, latencia y conteo de predicciones.
- Detectar entradas atípicas que podrían indicar cambios en el comportamiento de los datos.
- Garantizar la calidad del servicio mediante pruebas automatizadas.

### Alcance académico

El proyecto utiliza datos sintéticos generados por reglas determinísticas con semilla fija (`seed=42`), lo que garantiza reproducibilidad pero no representa un dataset de producción real. Las métricas obtenidas son válidas dentro del contexto académico.

---

## 2. Mejora técnica incorporada

A lo largo del desarrollo se incorporaron las siguientes mejoras de manera progresiva:

### 2.1 Cambio de hiperparámetro

El solver del clasificador se cambió de `lbfgs` (valor por defecto) a `liblinear`.

```python
LogisticRegression(solver="liblinear", random_state=42)
```

**Justificación:** `liblinear` está optimizado para datasets pequeños y es el único solver de scikit-learn que soporta penalización L1, lo que permite selección automática de variables en iteraciones futuras.

### 2.2 Incorporación de ROC-AUC como métrica de evaluación

Se agregó `roc_auc_score` al script `evaluar_modelo.py`, complementando las métricas existentes (accuracy, precision, recall, F1).

**Justificación:** En datasets con clases desbalanceadas, el accuracy puede ser engañoso. ROC-AUC evalúa la capacidad del modelo para distinguir entre clases en todos los umbrales posibles, independientemente del umbral de decisión.

| Métrica   | Valor  |
|-----------|-------:|
| Accuracy  | 0.8250 |
| F1-score  | 0.8659 |
| ROC-AUC   | 0.8749 |

### 2.3 Validación cruzada entre campos

Se incorporó una regla de coherencia en el modelo Pydantic de entrada:

> Un cliente con `antiguedad = 0` (cliente nuevo) no puede registrar `reclamos > 0`, ya que no ha tenido tiempo de generar quejas históricas.

```python
@model_validator(mode="after")
def validar_coherencia(self) -> "ClienteEntrada":
    if self.antiguedad == 0 and self.reclamos > 0:
        raise ValueError(
            "Un cliente nuevo (antiguedad=0) no puede tener reclamos previos."
        )
    return self
```

La API devuelve HTTP 422 cuando esta regla se viola, con un mensaje descriptivo del error.

### 2.4 Pruebas automatizadas para el endpoint de predicción

Se amplió la suite de tests de 2 a 13 casos, cubriendo:

- Estructura completa de la respuesta.
- Lógica de clasificación (bajo riesgo / alto riesgo).
- Detección de anomalías.
- Rechazo de entradas inválidas (campos negativos, campos faltantes).
- Validación cruzada entre campos.

### 2.5 Corrección de inconsistencias en el pipeline

Se identificaron y corrigieron tres inconsistencias estructurales:

| Inconsistencia | Causa | Corrección aplicada |
|---|---|---|
| `evaluar_modelo.py` cargaba modelo legacy | Ruta apuntaba a `modelo_churn.pkl` | Actualizado a `modelo_churn_v1.joblib` |
| `evaluar_modelo.py` usaba columnas incompatibles | `test.csv` tiene variables distintas al modelo activo | Reemplazado por función que reproduce el mismo split de entrenamiento |
| `test_api.py` verificaba clave inexistente | Clave `modelo_disponible` fue renombrada a `modelo` | Actualizado el assert correspondiente |

---

## 3. API y endpoints

La API está construida con **FastAPI** y expuesta mediante **Uvicorn**. Implementa un middleware de latencia y contadores thread-safe en memoria.

### Endpoints disponibles

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Verificación de actividad del servicio |
| `GET` | `/health` | Estado del servicio y versión del modelo cargado |
| `GET` | `/metrics` | Resumen acumulado de solicitudes, errores y latencia |
| `POST` | `/predict` | Predicción de churn para un cliente |

### Estructura de entrada — `POST /predict`

```json
{
  "antiguedad": 48,
  "cargo_mensual": 55.0,
  "reclamos": 0
}
```

| Campo | Tipo | Rango técnico | Rango histórico |
|---|---|---|---|
| `antiguedad` | entero | 0 – 240 meses | 1 – 72 meses |
| `cargo_mensual` | decimal | 0 – 5000 | 20.0 – 150.0 |
| `reclamos` | entero | 0 – 100 | 0 – 7 |

### Estructura de respuesta

```json
{
  "prediccion": "bajo_riesgo",
  "probabilidad": 0.0725,
  "version_modelo": "modelo_churn_v1",
  "autor": "Rodny Hurtado",
  "alertas_datos": []
}
```

El campo `alertas_datos` contiene una lista de advertencias cuando algún valor supera el rango histórico de entrenamiento, sin rechazar la solicitud.

### Validaciones activas

- Rangos de campo: valores negativos o excesivamente altos producen HTTP 422.
- Coherencia entre campos: `antiguedad=0` con `reclamos>0` produce HTTP 422.
- Disponibilidad del modelo: si el modelo no está cargado, `POST /predict` devuelve HTTP 503.

---

## 4. Ejecución dentro de Docker

El proyecto incluye un `Dockerfile` que empaqueta la API y sus dependencias en un contenedor reproducible.

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Pasos para construir y ejecutar

```bash
# Construir la imagen
docker build -t churn-api .

# Ejecutar el contenedor
docker run -p 8000:8000 churn-api
```

### Consideraciones de la imagen

- Se usa `python:3.12-slim` para minimizar el tamaño de la imagen.
- `--no-cache-dir` evita almacenar paquetes innecesarios dentro del contenedor.
- `--host 0.0.0.0` es obligatorio dentro de Docker para que el puerto sea accesible desde el host.
- Las métricas en memoria se reinician cada vez que el contenedor se detiene, ya que no hay persistencia externa.

### Verificación del servicio

```bash
curl http://localhost:8000/health
```

Respuesta esperada:
```json
{"estado": "ok", "modelo": "modelo_churn_v1"}
```

---

## 5. Propuesta de monitoreo

### Monitoreo implementado (nivel básico)

La API incorpora una primera capa de observabilidad directamente en `api/main.py`:

**Logging estructurado**
- Cada solicitud, predicción y error queda registrado en `logs/monitor_api.log` y en consola.
- Formato: `fecha y hora | nivel | mensaje`.

**Middleware de latencia**
- Mide el tiempo de procesamiento de cada solicitud con `perf_counter`.
- El valor se expone en el encabezado de respuesta `X-Process-Time-ms`.

**Contadores en memoria**
- Solicitudes totales, errores de validación, errores internos.
- Predicciones de alto y bajo riesgo.
- Solicitudes con anomalías en los datos de entrada.
- Latencia promedio y máxima acumuladas.

**Detección de anomalías por rango**
- Compara cada campo de entrada contra los rangos históricos definidos en `RANGOS_HISTORICOS`.
- Cuando un valor se aparta del rango histórico, se registra un aviso y se incluye en `alertas_datos`.

### Propuesta de evolución (siguiente etapa)

Para un entorno de producción, se propone extender el monitoreo con las siguientes herramientas:

| Herramienta | Rol |
|---|---|
| **MLflow** | Registro de experimentos, métricas por versión de modelo y comparación de runs |
| **Prometheus** | Exposición de métricas en formato estándar para scraping |
| **Grafana** | Visualización de dashboards: latencia, tasa de errores, distribución de predicciones |
| **Alertmanager** | Notificaciones automáticas cuando métricas superan umbrales definidos |

**Alertas críticas sugeridas:**

- Tasa de error (`errores_validacion / solicitudes_totales`) > 10 %
- Latencia promedio > 500 ms
- Proporción de predicciones `alto_riesgo` > 70 % (posible drift)
- Solicitudes con anomalías > 20 % del total

---

## 6. Error o incidente

### Incidente registrado: WinError 10013

**Descripción**

Al intentar iniciar la API con el comando:

```bash
python -m uvicorn api.main:app --reload
```

Se produjo el siguiente error en Windows:

```
ERROR: [WinError 10013] An attempt was made to access a socket
in a way forbidden by its access permissions
```

**Causa**

El puerto 8000 estaba bloqueado por uno de los siguientes motivos:
- Una instancia anterior de Uvicorn seguía corriendo en segundo plano.
- El firewall de Windows o un antivirus bloqueaba el acceso al puerto.
- El proceso no tenía permisos suficientes para enlazarse al puerto.

**Resolución aplicada**

Se cambió el puerto de inicio para evitar el conflicto:

```bash
python -m uvicorn api.main:app --reload --port 8001
```

**Resolución alternativa (diagnóstico completo)**

```bash
# Identificar qué proceso ocupa el puerto 8000
netstat -ano | findstr :8000

# Terminar el proceso con el PID encontrado
taskkill /PID <PID> /F

# Reiniciar la API en el puerto original
python -m uvicorn api.main:app --reload
```

**Lección operativa**

En entornos de desarrollo en Windows, el uso de `--reload` puede dejar procesos huérfanos si la terminal se cierra de forma abrupta. Es recomendable verificar los puertos ocupados antes de iniciar el servicio, o estandarizar el uso de Docker para aislar el proceso del sistema operativo anfitrión.

---

## 7. Drift y respuesta operativa

### Definición de drift en este contexto

El *data drift* ocurre cuando la distribución de los datos de entrada en producción se aleja de la distribución de los datos con los que el modelo fue entrenado. Esto puede degradar la calidad de las predicciones sin que los errores técnicos sean visibles.

### Mecanismo de detección implementado

La API compara cada solicitud entrante contra los rangos históricos registrados en el momento del entrenamiento:

```python
RANGOS_HISTORICOS = {
    "antiguedad": (1, 72),
    "cargo_mensual": (20.0, 150.0),
    "reclamos": (0, 7),
}
```

Cuando un valor supera estos límites, se genera una alerta en `alertas_datos` y se incrementa el contador `solicitudes_con_anomalias`. La solicitud no es rechazada — el modelo responde, pero queda registrado el evento.

### Escenario simulado

**Entrada atípica:**

```json
{
  "antiguedad": 180,
  "cargo_mensual": 600.0,
  "reclamos": 35
}
```

**Respuesta:**

```json
{
  "prediccion": "alto_riesgo",
  "probabilidad": 1.0,
  "alertas_datos": [
    "antiguedad=180 fuera del rango histórico [1, 72]",
    "cargo_mensual=600.0 fuera del rango histórico [20.0, 150.0]",
    "reclamos=35 fuera del rango histórico [0, 7]"
  ]
}
```

### Protocolo de respuesta operativa ante drift

Si el porcentaje de solicitudes con anomalías supera el umbral definido (20 %), se propone el siguiente protocolo:

```
1. DETECTAR    → Alerta automática por proporción de anomalías
2. ANALIZAR    → Revisar distribución de los campos afectados
3. DECIDIR     → ¿Es ruido puntual o cambio estructural?
4. RESPONDER   → Si es cambio estructural:
                   a. Recolectar nuevos datos representativos
                   b. Reentrenar el modelo con datos actualizados
                   c. Actualizar RANGOS_HISTORICOS
                   d. Desplegar nueva versión con versionado explícito
5. DOCUMENTAR  → Registrar el incidente y la acción tomada
```

---

## 8. Conclusión

El proyecto demostró la viabilidad de construir un pipeline de MLOps funcional con herramientas de código abierto y complejidad controlada. A lo largo del desarrollo se aplicaron prácticas fundamentales:

**Lo que se logró:**

- Pipeline completo: generación de datos → entrenamiento → evaluación → servicio HTTP.
- Modelo con métricas reales (Accuracy 0.825, ROC-AUC 0.875) sobre 800 registros sintéticos.
- API con validación de entradas en dos niveles: campos individuales y coherencia cruzada.
- Observabilidad básica: logging estructurado, latencia, contadores y detección de anomalías.
- Suite de pruebas automatizadas con 13 casos que cubren el camino feliz y los casos borde.
- Contenerización con Docker para reproducibilidad del entorno.

**Limitaciones identificadas:**

- Los datos son sintéticos. Un modelo productivo requiere datos reales con volumen suficiente.
- Las métricas en memoria se pierden al reiniciar el servicio. Se necesita persistencia externa.
- No existe CI/CD automatizado. El despliegue es manual.
- No hay registro formal de experimentos (MLflow u equivalente).

**Próximos pasos recomendados:**

1. Integrar MLflow para trazabilidad de experimentos y registro de modelos.
2. Configurar variables de entorno para desacoplar la configuración del código.
3. Implementar un pipeline de CI/CD con GitHub Actions.
4. Conectar Prometheus y Grafana para monitoreo persistente en producción.
