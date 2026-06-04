# Descripción del dataset

Este proyecto utiliza un dataset de ejemplo para analizar el abandono de clientes, conocido como churn.

## Variable objetivo

La variable objetivo es:

```text
churn
```

Valores posibles:

- `0`: el cliente no abandonó el servicio.
- `1`: el cliente abandonó el servicio.

## Variables predictoras

El dataset utiliza las siguientes variables:

| Variable | Descripción |
|---|---|
| edad | Edad del cliente. |
| antiguedad_meses | Tiempo de permanencia del cliente en meses. |
| saldo_promedio | Saldo promedio del cliente. |
| reclamos | Cantidad de reclamos realizados. |
| usa_app | Indica si el cliente usa la aplicación móvil. |

## Archivos que se generarán

Durante la práctica se generarán los siguientes archivos:

```text
data/churn_clientes.csv
data/train.csv
data/test.csv
```

Estos archivos serán creados automáticamente por el script:

```text
src/preparar_datos.py
```