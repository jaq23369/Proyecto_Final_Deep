# Costo de inferencia por modelo

Generado por `python scripts/inference_costs.py`. Cada modelo se midió con `cupo.benchmark` (mismo protocolo, uno tras otro en la misma máquina: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz) con 46 patches de val (semilla 0), 300 repeticiones por espacio y 100 por frame. Los JSON de cada medición están en `results/costo_inferencia/`.

Los modelos se midieron en la misma sesión para que la comparación entre ellos sea justa. El valor absoluto de la latencia y del costo cambia entre sesiones y entre máquinas (por ejemplo, `models/benchmark.json` del modelo final se midió en otra máquina); la diferencia entre modelos y la conclusión frente a la referencia son lo que se compara.

## Latencia y costo de inferencia

| Modelo | Parámetros | Archivo (MB) | CPU 1 hilo, espacio p50 (ms) | CPU 1 hilo, frame de 46 p50 (ms) | CPU todos los hilos, frame p50 (ms) | Throughput 1 núcleo con preproc. (pred/s) | Costo por 1,000 pred. (Q) | Costo de inferencia (Q/mes) | GPU, frame p50 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| B1 (b1_hog_lr) | 1,765 | 0.049 | 2.05 | 70.6 | 70.9 | 489 | 0.000219 | 1.81 | — |
| B2 (b2_malexnet) | 42,460 | 0.191 | 2.19 | 91.4 | 30.1 | 254 | 0.000421 | 3.48 | 10.30 |
| Modelo final (mnv3_ft_cost) | 1,519,906 | 5.807 | 2.86 | 137.6 | 62.0 | 219 | 0.000489 | 4.04 | 18.52 |

- La latencia por espacio es batch 1; la del frame, un batch de 46 espacios. CPU con ONNX Runtime (B2 y modelo final) o con scikit-image + scikit-learn (B1).
- En B1 la latencia del modelo incluye el cálculo de HOG (su extracción de características: gris, 64x64 y `skimage.feature.hog`), con el mismo camino que la app; su preprocesamiento es solo abrir y decodificar la imagen. En las CNN el preprocesamiento (`preprocess_numpy`) se mide aparte.
- Costo por 1,000 predicciones: tiempo de 1 núcleo en modo frame, modelo + preprocesamiento, a USD 0.05/h por núcleo y Q7.70 por USD (`configs/costs.yaml`).
- Costo de inferencia al mes: 8,265,600 predicciones al mes del escenario comercial (`configs/value/comercial.yaml`: espacios · 3600/refresh_s · horas de operación · días) × costo por 1,000 / 1,000, con `value_model.inference_cost_period`.
- GPU: PyTorch con CUDA (AMP si el run lo usaba); solo para los modelos de PyTorch con su run local.

## Valor de negocio frente al costo

| Modelo | Valor val (Q/mes) | Valor test (Q/mes) | Valor OOD (Q/mes) | Costo de entrenamiento (Q) | Δ valor val vs B1 (b1_hog_lr) | Δ valor test | Δ valor OOD | Δ costo de inferencia (Q/mes) | Δ neto val (Q/mes) |
|---|---|---|---|---|---|---|---|---|---|
| B1 (b1_hog_lr) | 53,388.86 | 52,175.87 | 45,729.63 | 0.008 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| B2 (b2_malexnet) | 53,741.00 | 53,801.90 | 47,926.57 | 0.673 | +352.14 | +1,626.03 | +2,196.93 | +1.67 | +350.47 |
| Modelo final (mnv3_ft_cost) | 53,982.04 | 53,946.11 | 52,238.41 | 0.578 | +593.18 | +1,770.24 | +6,508.77 | +2.24 | +590.95 |

- Valor comercial (Q/mes para el operador), parámetros y costo de entrenamiento: de `results/iteraciones.csv`. Test y OOD se evaluaron una sola vez, después del congelamiento.
- Ese valor descuenta para todos los modelos el mismo costo de inferencia por defecto (`costo_por_1000_pred_Q` de `configs/costs.yaml`), así que Δ valor no incluye la diferencia de cómputo; Δ neto val = Δ valor val − Δ costo de inferencia medido.

## Conclusión

- B2 (b2_malexnet) genera al mes, frente a B1 (b1_hog_lr), Q352 más en validación, Q1,626 más en test y Q2,197 más en otro parqueo (CNRPark A/B, OOD), con un costo extra de cómputo de Q1.67 al mes (inferencia en CPU con 1 núcleo para 8,265,600 predicciones al mes del escenario comercial). En validación, descontado ese cómputo, la diferencia neta es de Q350 al mes a favor del modelo.

- Modelo final (mnv3_ft_cost) genera al mes, frente a B1 (b1_hog_lr), Q593 más en validación, Q1,770 más en test y Q6,509 más en otro parqueo (CNRPark A/B, OOD), con un costo extra de cómputo de Q2.24 al mes (inferencia en CPU con 1 núcleo para 8,265,600 predicciones al mes del escenario comercial). En validación, descontado ese cómputo, la diferencia neta es de Q591 al mes a favor del modelo.
