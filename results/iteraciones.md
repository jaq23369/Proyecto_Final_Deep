# Tabla de iteraciones

Generada con `scripts/results_table.py` a partir de 2 run(s) en `runs/`.

## Validación

| ID | Modelo | Cambios | Semillas | Train acc | Val F1 macro | Val ocupado→libre | Val valor comercial (Q/mes) [IC95] | Δ valor vs. iteración anterior (Q/mes) | Valor comercial sin franja, mejor umbral único (Q/mes) | Operadores extra (comercial) | Val horas UVG (h/semestre) | Val valor UVG (Q/semestre) | Valor UVG sin franja, mejor umbral único (Q/semestre) | Operadores extra (UVG) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B0** (b0_status_quo) | sin sistema | Sin sistema: el conductor busca como hoy (T_sys = t_base); ahorro 0 y sin costos | – | – | – | – | 0 | – | 0 | 0.00 | 0 | 0 | 0 | 0.00 |
| **B1** (b1_hog_lr) | hog_lr | HOG 64x64 en gris + StandardScaler + regresión logística; C elegido en val por valor | 42 | 0.9636 | 0.9490 | 4.76 % | 53,389 [53,137; 53,483] | +53,389 (vs. B0) | 52,857 (t = 0.04) | 0.00 | 7,881 | 161,522 [160,777; 161,801] | 159,025 (t = 0.04) | 0.00 |
| **B2** (b2_malexnet) | malexnet | CNN mAlexNet desde cero, sin aumento de datos y con lr constante (baseline de deep learning) | 42 | 0.9817 | 0.9787 | 0.84 % | 53,741 [53,455; 53,798] | +352 (vs. B1) | 53,484 (t = 0.09) | 0.00 | 7,938 | 162,712 [162,006; 162,897] | 161,552 (t = 0.09) | 0.00 |

## Costo y operación

| ID | Params | Tamaño (MB) | Latencia CPU p50 (ms/espacio) | Costo de inferencia (Q/mes, comercial) | Tiempo de entrenamiento (total) | Costo de entrenamiento total (Q) | Calibrado (T) | t_L / t_H comercial | Por verificar (val, comercial) | t_L / t_H UVG |
|---|---|---|---|---|---|---|---|---|---|---|
| **B1** (b1_hog_lr) | 1.8 k | 0.05 | – | 165 (costs.yaml) | 71 s | 0.008 | no | 0.02 / 0.09 | 6.07 % | 0.01 / 0.07 |
| **B2** (b2_malexnet) | 42.5 k | 0.17 | – | 165 (costs.yaml) | 15.0 min | 0.673 | no | 0.03 / 0.58 | 6.09 % | 0.01 / 0.38 |

## ¿Cambia el ranking con los supuestos?

Cada modelo conserva sus tasas en val y su punto de operación; se mueve un supuesto a la vez. Detalle en `ranking_sensibilidad_<escenario>.csv`.

- **Comercial**: en el caso base gana **b2_malexnet**. El ganador cambia en 0 de 104 variaciones (cada supuesto ×0.5, ×0.8, ×1.2, ×1.5) y el orden completo cambia en 0 de 104.
- **UVG**: en el caso base gana **b2_malexnet**. El ganador cambia en 2 de 108 variaciones (cada supuesto ×0.5, ×0.8, ×1.2, ×1.5) y el orden completo cambia en 2 de 108: Ocupación en hora pico (π) ×0.5 → b1_hog_lr; Ocupación en hora pico (π) ×0.8 → b1_hog_lr.

## Notas

- **B0** es el status quo (sin sistema): fila sintética con valor 0 en ambos escenarios.
- Con varias semillas cada celda es media ± desviación estándar; el IC95 es el promedio de los IC95 bootstrap de cada semilla. Tiempo y costo de entrenamiento se suman.
- **Valor**: comercial = valor para el operador (Q/mes); UVG = valor neto del tiempo ahorrado (Q/semestre). Se usan los umbrales (t_L, t_H) óptimos en val; en runs calibrados, la temperatura y los umbrales de `calibration.json`.
- **Δ valor**: diferencia frente a la mejor fila comparable (mismo `subset`; B0 vale para todas) de la iteración anterior presente, en el orden B0, B1, B2, K1…K6.
- **Sin franja, mejor umbral único**: decisión de 2 estados (t_L = t_H = t) con el t ∈ [0, 1] que maximiza el valor en val. **Operadores extra**: personal adicional (tiempo completo equivalente) para revisar los espacios por verificar en el punto de operación.
- **Costo de inferencia**: predicciones del mes del escenario comercial × costo por 1,000 predicciones (de `benchmark.json` si existe; si no, el de `configs/costs.yaml`).
- **Train acc**: en los modelos de DL es la media de la época (con aumento y dropout); en B1 es la accuracy del modelo final sobre train.
- Ocupado→libre: P(mostrado libre | ocupado) con umbral 0.5, el error más caro.
