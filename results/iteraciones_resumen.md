# Resumen de iteraciones

Una fila por iteración (si hay variantes, la de mayor valor comercial en val). Test solo aparece tras la evaluación única del congelamiento.

| Iteración | Modelo | Train acc | Val F1 macro | Val valor comercial (Q/mes) | Test F1 macro | Test valor comercial (Q/mes) | Latencia CPU p50 (ms/espacio) | Costo de entrenamiento total (Q) |
|---|---|---|---|---|---|---|---|---|
| **B0** (b0_status_quo) | sin sistema | – | – | 0 | – | – | – | – |
| **B1** (b1_hog_lr) | hog_lr | 0.9636 | 0.9490 | 53,389 | – | – | – | 0.008 |
| **B2** (b2_malexnet) | malexnet | 0.9817 | 0.9787 | 53,741 | – | – | – | 0.673 |
