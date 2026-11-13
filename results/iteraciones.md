# Tabla de iteraciones

Generada con `scripts/results_table.py` a partir de 10 run(s) en `runs/`.

## Validación

| ID | Modelo | Cambios | Semillas | Train acc | Val F1 macro | Val ocupado→libre | Val valor comercial (Q/mes) [IC95] | Δ valor vs. iteración anterior (Q/mes) | Valor comercial sin franja, mejor umbral único (Q/mes) | Operadores extra (comercial) | Val horas UVG (h/semestre) | Val valor UVG (Q/semestre) | Valor UVG sin franja, mejor umbral único (Q/semestre) | Operadores extra (UVG) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B0** (b0_status_quo) | sin sistema | Sin sistema: el conductor busca como hoy (T_sys = t_base); ahorro 0 y sin costos | – | – | – | – | 0 | – | 0 | 0.00 | 0 | 0 | 0 | 0.00 |
| **B1** (b1_hog_lr) | hog_lr | HOG 64x64 en gris + StandardScaler + regresión logística; C elegido en val por valor | 42 | 0.9636 | 0.9490 | 4.76 % | 53,389 [53,137; 53,483] | +53,389 (vs. B0) | 52,857 (t = 0.04) | 0.00 | 7,881 | 161,522 [160,777; 161,801] | 159,025 (t = 0.04) | 0.00 |
| **B2** (b2_malexnet) | malexnet | CNN mAlexNet desde cero, sin aumento de datos y con lr constante (baseline de deep learning) | 42 | 0.9817 | 0.9787 | 0.84 % | 53,741 [53,455; 53,798] | +352 (vs. B1) | 53,484 (t = 0.09) | 0.00 | 7,938 | 162,712 [162,006; 162,897] | 161,552 (t = 0.09) | 0.00 |
| **malexnet_aug** | malexnet | B2 + aumento de datos (iluminación, blur, oclusiones) + scheduler coseno con 1 época de warmup (B2 usa lr constante) | 42 | 0.9814 | 0.9831 | 1.02 % | 53,825 [53,674; 53,868] | +84 (vs. B2) | 53,563 (t = 0.05) | 0.00 | 7,951 | 162,991 [162,366; 163,121] | 161,874 (t = 0.05) | 0.00 |
| **mnv3_head** | mobilenet_v3_small | MobileNetV3-Small de ImageNet con el backbone congelado; solo se entrena la cabeza | 42 | 0.9971 | 0.9926 | 0.42 % | 53,913 [53,868; 53,952] | +88 (vs. malexnet_aug) | 53,831 (t = 0.03) | 0.00 | 7,965 | 163,276 [163,092; 163,435] | 162,950 (t = 0.03) | 0.00 |
| **mnv3_ft** | mobilenet_v3_small | mnv3_head con fine-tuning completo (todo el backbone), lr 3e-4 coseno con warmup y aumento de datos | 42 | 0.9982 | 0.9958 | 0.26 % | 53,955 [53,919; 53,986] | +42 (vs. mnv3_head) | 53,919 (t = 0.03) | 0.00 | 7,973 | 163,445 [163,296; 163,571] | 163,300 (t = 0.03) | 0.00 |
| **mnv3_ft_cost** | mobilenet_v3_small | mnv3_ft + pérdida sensible al costo: peso 3 a la clase ocupado para castigar ocupado→libre | 42 | 0.9961 | 0.9955 | 0.33 % | 53,982 [53,966; 53,993] | +27 (vs. mnv3_ft) | 53,916 (t = 0.08) | 0.00 | 7,978 | 163,557 [163,490; 163,601] | 163,288 (t = 0.03) | 0.00 |
| **efficientnet_b0** | efficientnet_b0 | EfficientNet-B0 de ImageNet con fine-tuning completo y aumento (modelo grande de referencia) | 42 | 0.9947 | 0.9929 | 0.64 % | 53,951 [53,915; 53,978] | -31 (vs. mnv3_ft_cost) | 53,857 (t = 0.03) | 0.00 | 7,972 | 163,429 [163,283; 163,541] | 163,054 (t = 0.03) | 0.00 |
| **mnv3_ft_seeds** | mobilenet_v3_small | mnv3_ft_cost (mnv3_ft con pérdida sensible al costo) con 3 semillas (42, 43, 44) + calibración por temperatura; incertidumbre entre semillas | 42, 43, 44 | 0.9959 ± 0.0020 | 0.9939 ± 0.0020 | 0.20 % ± 0.13 % | 53,972 ± 6 [53,949; 53,989] | +21 (vs. efficientnet_b0) | 53,907 ± 27 (t = 0.26 ± 0.27) | 0.00 ± 0.00 | 7,976 ± 1 | 163,516 ± 26 [163,421; 163,585] | 163,252 ± 109 (t = 0.26 ± 0.27) | 0.00 ± 0.00 |

## Costo y operación

| ID | Params | Tamaño (MB) | Latencia CPU p50 (ms/espacio) | Costo de inferencia (Q/mes, comercial) | Tiempo de entrenamiento (total) | Costo de entrenamiento total (Q) | Calibrado (T) | t_L / t_H comercial | Por verificar (val, comercial) | t_L / t_H UVG |
|---|---|---|---|---|---|---|---|---|---|---|
| **B1** (b1_hog_lr) | 1.8 k | 0.05 | – | 165 (costs.yaml) | 71 s | 0.008 | no | 0.02 / 0.09 | 6.07 % | 0.01 / 0.07 |
| **B2** (b2_malexnet) | 42.5 k | 0.17 | – | 165 (costs.yaml) | 15.0 min | 0.673 | no | 0.03 / 0.58 | 6.09 % | 0.01 / 0.38 |
| **malexnet_aug** | 42.5 k | 0.17 | – | 165 (costs.yaml) | 29.5 min | 1.326 | no | 0.02 / 0.43 | 6.03 % | 0.01 / 0.24 |
| **mnv3_head** | 1.52 M | 5.93 | – | 165 (costs.yaml) | 7.9 min | 0.355 | no | 0.01 / 0.45 | 2.25 % | 0.01 / 0.45 |
| **mnv3_ft** | 1.52 M | 5.93 | – | 165 (costs.yaml) | 22.4 min | 1.006 | no | 0.01 / 0.27 | 1.34 % | 0.01 / 0.27 |
| **mnv3_ft_cost** | 1.52 M | 5.93 | 1.85 | 3 (benchmark.json) | 12.9 min | 0.578 | no | 0.01 / 0.23 | 2.77 % | 0.01 / 0.23 |
| **efficientnet_b0** | 4.01 M | 15.58 | – | 165 (costs.yaml) | 14.4 min | 0.648 | no | 0.01 / 0.18 | 2.20 % | 0.01 / 0.18 |
| **mnv3_ft_seeds** | 1.52 M | 5.93 ± 0.00 | – | 165 ± 0 (costs.yaml) | 40.1 min (3 runs; 13.4 min c/u) | 1.800 (3 runs; 0.600 c/u) | sí (T = 1.252 ± 0.206) | 0.08 / 0.65 (media) | 2.68 % ± 1.33 % | 0.08 / 0.65 (media) |

## Test y fuera de dominio (evaluación única tras el congelamiento)

Umbrales y temperatura elegidos en val; nunca se re-optimizan en test ni en OOD.

| ID | Test F1 macro | Test ocupado→libre | Test valor comercial (Q/mes) | OOD (CNRPark A/B) F1 macro | OOD (CNRPark A/B) ocupado→libre | OOD (CNRPark A/B) valor comercial (Q/mes) |
|---|---|---|---|---|---|---|
| **B1** (b1_hog_lr) | 0.9556 | 4.73 % | 52,176 | 0.8396 | 16.10 % | 45,730 |
| **B2** (b2_malexnet) | 0.9826 | 0.95 % | 53,802 | 0.8984 | 2.64 % | 47,927 |
| **malexnet_aug** | 0.9874 | 0.84 % | 53,783 | 0.9277 | 2.81 % | 50,282 |
| **mnv3_head** | 0.9956 | 0.31 % | 53,942 | 0.9409 | 5.61 % | 49,972 |
| **mnv3_ft** | 0.9976 | 0.27 % | 53,929 | 0.9536 | 3.53 % | 52,616 |
| **mnv3_ft_cost** | 0.9973 | 0.27 % | 53,946 | 0.9506 | 4.03 % | 52,238 |
| **efficientnet_b0** | 0.9971 | 0.40 % | 53,879 | 0.9572 | 4.92 % | 53,599 |
| **mnv3_ft_seeds** | 0.9957 ± 0.0026 | 0.21 % ± 0.07 % | 53,933 ± 19 | 0.9445 ± 0.0135 | 3.40 % ± 0.67 % | 51,000 ± 860 |

## ¿Cambia el ranking con los supuestos?

Cada modelo conserva sus tasas en val y su punto de operación; se mueve un supuesto a la vez. Detalle en `ranking_sensibilidad_<escenario>.csv`.

- **Comercial**: en el caso base gana **mnv3_ft_cost**. El ganador cambia en 4 de 104 variaciones (cada supuesto ×0.5, ×0.8, ×1.2, ×1.5) y el orden completo cambia en 8 de 104: Ocupación en hora pico (π) ×0.5 → mnv3_ft; Precisión de la verificación (prob.) ×0.5 → mnv3_ft; Precisión de la verificación (prob.) ×1.2 → mnv3_ft_seeds; Precisión de la verificación (prob.) ×1.5 → mnv3_ft_seeds.
- **UVG**: en el caso base gana **mnv3_ft_cost**. El ganador cambia en 4 de 108 variaciones (cada supuesto ×0.5, ×0.8, ×1.2, ×1.5) y el orden completo cambia en 6 de 108: Ocupación en hora pico (π) ×0.5 → mnv3_ft; Precisión de la verificación (prob.) ×0.5 → mnv3_ft; Precisión de la verificación (prob.) ×1.2 → mnv3_ft_seeds; Precisión de la verificación (prob.) ×1.5 → mnv3_ft_seeds.

## Notas

- **B0** es el status quo (sin sistema): fila sintética con valor 0 en ambos escenarios.
- Con varias semillas cada celda es media ± desviación estándar; el IC95 es el promedio de los IC95 bootstrap de cada semilla. Tiempo y costo de entrenamiento se suman.
- **Valor**: comercial = valor para el operador (Q/mes); UVG = valor neto del tiempo ahorrado (Q/semestre). Se usan los umbrales (t_L, t_H) óptimos en val; en runs calibrados, la temperatura y los umbrales de `calibration.json`.
- **Δ valor**: diferencia frente a la mejor fila comparable (mismo `subset`; B0 vale para todas) de la iteración anterior presente, en el orden B0, B1, B2, malexnet_aug…mnv3_ft_seeds.
- **Sin franja, mejor umbral único**: decisión de 2 estados (t_L = t_H = t) con el t ∈ [0, 1] que maximiza el valor en val. **Operadores extra**: personal adicional (tiempo completo equivalente) para revisar los espacios por verificar en el punto de operación.
- **Costo de inferencia**: predicciones del mes del escenario comercial × costo por 1,000 predicciones (de `benchmark.json` si existe; si no, el de `configs/costs.yaml`).
- **Train acc**: en los modelos de DL es la media de la época (con aumento y dropout); en B1 es la accuracy del modelo final sobre train.
- Ocupado→libre: P(mostrado libre | ocupado) con umbral 0.5, el error más caro.
