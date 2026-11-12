# Resumen de iteraciones

Una fila por iteración (si hay variantes, la de mayor valor comercial en val). Test solo aparece tras la evaluación única del congelamiento.

| Iteración | Modelo | Train acc | Val F1 macro | Val valor comercial (Q/mes) | Test F1 macro | Test valor comercial (Q/mes) | Latencia CPU p50 (ms/espacio) | Costo de entrenamiento total (Q) |
|---|---|---|---|---|---|---|---|---|
| **B0** (b0_status_quo) | sin sistema | – | – | 0 | – | – | – | – |
| **B1** (b1_hog_lr) | hog_lr | 0.9636 | 0.9490 | 53,389 | – | – | – | 0.008 |
| **B2** (b2_malexnet) | malexnet | 0.9817 | 0.9787 | 53,741 | – | – | – | 0.673 |
| **malexnet_aug** | malexnet | 0.9814 | 0.9831 | 53,825 | – | – | – | 1.326 |
| **mnv3_head** | mobilenet_v3_small | 0.9971 | 0.9926 | 53,913 | – | – | – | 0.355 |
| **mnv3_ft** | mobilenet_v3_small | 0.9982 | 0.9958 | 53,955 | – | – | – | 1.006 |
| **mnv3_ft_cost** | mobilenet_v3_small | 0.9961 | 0.9955 | 53,982 | – | – | 1.85 | 0.578 |
| **efficientnet_b0** | efficientnet_b0 | 0.9947 | 0.9929 | 53,951 | – | – | – | 0.648 |
| **mnv3_ft_seeds** | mobilenet_v3_small | 0.9959 ± 0.0020 | 0.9939 ± 0.0020 | 53,972 ± 6 | – | – | – | 1.800 |
