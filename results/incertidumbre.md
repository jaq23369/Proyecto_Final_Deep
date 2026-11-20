# Incertidumbre en test y OOD

Generada con `scripts/uncertainty.py` a partir de las predicciones congeladas de cada run. IC95 por bootstrap (1,000 réplicas, semilla 0).

## Test: CNR-EXT, fechas de test (n = 31,825)

| ID | F1 macro [IC95] | Ocupado→libre [IC95] | Valor comercial (Q/mes) [IC95] | Valor UVG (Q/semestre) [IC95] |
|---|---|---|---|---|
| **B1** (b1_hog_lr) | 0.9556 [0.9533; 0.9578] | 4.73 % [4.43 %; 5.03 %] | 52,176 [51,917; 52,421] | 158,292 [157,544; 159,027] |
| **B2** (b2_malexnet) | 0.9826 [0.9812; 0.9840] | 0.95 % [0.82 %; 1.08 %] | 53,802 [53,753; 53,845] | 162,799 [162,181; 163,018] |
| **malexnet_aug** | 0.9874 [0.9863; 0.9886] | 0.84 % [0.72 %; 0.98 %] | 53,783 [53,581; 53,887] | 163,057 [162,906; 163,171] |
| **mnv3_head** | 0.9956 [0.9949; 0.9963] | 0.31 % [0.24 %; 0.39 %] | 53,942 [53,909; 53,971] | 163,394 [163,258; 163,510] |
| **mnv3_ft** | 0.9976 [0.9970; 0.9981] | 0.27 % [0.20 %; 0.34 %] | 53,929 [53,890; 53,965] | 163,338 [163,177; 163,486] |
| **mnv3_ft_cost** — modelo final | 0.9973 [0.9967; 0.9978] | 0.27 % [0.20 %; 0.35 %] | 53,946 [53,910; 53,974] | 163,409 [163,262; 163,523] |
| **efficientnet_b0** | 0.9971 [0.9966; 0.9977] | 0.40 % [0.32 %; 0.50 %] | 53,879 [53,821; 53,925] | 163,132 [162,898; 163,320] |
| **mnv3_ft_seeds** | 0.9957 ± 0.0026 [0.9950; 0.9964] | 0.21 % ± 0.07 % [0.15 %; 0.28 %] | 53,933 ± 19 [53,893; 53,965] | 163,354 ± 76 [163,191; 163,488] |

## OOD: CNRPark A/B (n = 12,584)

| ID | F1 macro [IC95] | Ocupado→libre [IC95] | Valor comercial (Q/mes) [IC95] | Valor UVG (Q/semestre) [IC95] |
|---|---|---|---|---|
| **B1** (b1_hog_lr) | 0.8396 [0.8331; 0.8461] | 16.10 % [15.29 %; 16.87 %] | 45,730 [45,041; 46,441] | 137,663 [135,238; 140,203] |
| **B2** (b2_malexnet) | 0.8984 [0.8927; 0.9047] | 2.64 % [2.31 %; 2.98 %] | 47,927 [47,399; 48,461] | 149,096 [147,517; 150,626] |
| **malexnet_aug** | 0.9277 [0.9230; 0.9326] | 2.81 % [2.48 %; 3.15 %] | 50,282 [49,774; 50,765] | 156,294 [154,935; 157,629] |
| **mnv3_head** | 0.9409 [0.9369; 0.9454] | 5.61 % [5.11 %; 6.08 %] | 49,972 [49,360; 50,597] | 151,472 [149,433; 153,501] |
| **mnv3_ft** | 0.9536 [0.9497; 0.9577] | 3.53 % [3.15 %; 3.92 %] | 52,616 [52,363; 52,852] | 157,984 [156,948; 158,949] |
| **mnv3_ft_cost** — modelo final | 0.9506 [0.9465; 0.9547] | 4.03 % [3.65 %; 4.45 %] | 52,238 [51,726; 52,732] | 160,832 [159,307; 162,035] |
| **efficientnet_b0** | 0.9572 [0.9538; 0.9608] | 4.92 % [4.47 %; 5.36 %] | 53,599 [53,291; 53,686] | 162,000 [161,574; 162,377] |
| **mnv3_ft_seeds** | 0.9445 ± 0.0135 [0.9401; 0.9487] | 3.40 % ± 0.67 % [3.04 %; 3.78 %] | 51,000 ± 860 [50,453; 51,521] | 155,733 ± 3,318 [154,136; 157,205] |

## Diferencia pareada: modelo final frente a los baselines

| Comparación | Partición | Δ valor comercial (Q/mes) [IC95] | Final > ref. (comercial) | Δ valor UVG (Q/semestre) [IC95] | Final > ref. (UVG) |
|---|---|---|---|---|---|
| mnv3_ft_cost − B1 | Test | +1,770 [+1,530; +2,037] | 100.0 % | +5,116 [+4,408; +5,863] | 100.0 % |
| mnv3_ft_cost − B1 | OOD | +6,509 [+5,747; +7,187] | 100.0 % | +23,169 [+20,857; +25,413] | 100.0 % |
| mnv3_ft_cost − B2 | Test | +144 [+98; +189] | 100.0 % | +610 [+378; +1,240] | 100.0 % |
| mnv3_ft_cost − B2 | OOD | +4,312 [+3,759; +4,824] | 100.0 % | +11,736 [+10,301; +12,934] | 100.0 % |

## Verificación

- 48 de 48 estimaciones puntuales (F1 macro, ocupado→libre y valor comercial en test y OOD; media y desviación en grupos con varias semillas) coinciden con `iteraciones.csv` (tolerancia relativa 1e-06).

## Conclusiones

- En OOD (CNRPark A/B), los IC95 del valor comercial del modelo final (mnv3_ft_cost) [Q51,726; Q52,732] y de B1 [Q45,041; Q46,441] no se solapan. La diferencia pareada es Δ = Q6,509 al mes [IC95 Q5,747; Q7,187], que excluye el 0; mnv3_ft_cost supera a B1 en 100.0 % de las remuestras.
- En OOD (CNRPark A/B), los IC95 del valor comercial del modelo final (mnv3_ft_cost) [Q51,726; Q52,732] y de B2 [Q47,399; Q48,461] no se solapan. La diferencia pareada es Δ = Q4,312 al mes [IC95 Q3,759; Q4,824], que excluye el 0; mnv3_ft_cost supera a B2 en 100.0 % de las remuestras.
- En test, los IC95 del valor comercial del modelo final (mnv3_ft_cost) [Q53,910; Q53,974] y de B1 [Q51,917; Q52,421] no se solapan. La diferencia pareada es Δ = Q1,770 al mes [IC95 Q1,530; Q2,037], que excluye el 0; mnv3_ft_cost supera a B1 en 100.0 % de las remuestras.
- En test, los IC95 del valor comercial del modelo final (mnv3_ft_cost) [Q53,910; Q53,974] y de B2 [Q53,753; Q53,845] no se solapan. La diferencia pareada es Δ = Q144 al mes [IC95 Q98; Q189], que excluye el 0; mnv3_ft_cost supera a B2 en 100.0 % de las remuestras.
- En OOD, mnv3_ft (Q52,616, IC95 solapado con el del final) y efficientnet_b0 (Q53,599, IC95 sin solaparse con el del final) superan en valor comercial puntual al modelo final (Q52,238). El final se eligió por valor en validación antes del congelamiento y no se cambia con estos números.
- mnv3_ft_seeds: en test la desviación entre semillas (Q19) es 0.5 veces la mitad del IC95 bootstrap (Q36); en OOD la desviación entre semillas (Q860) es 1.6 veces la mitad del IC95 bootstrap (Q534). Cuando ese cociente se acerca o supera 1, la variación por la semilla de entrenamiento pesa tanto como la del muestreo de imágenes.

## Notas

- Bootstrap de percentiles sobre imágenes (remuestreo con reemplazo), con la misma semilla y el mismo número de réplicas que `cupo.evaluate`; la estimación puntual es la de la muestra completa.
- Umbrales (t_L, t_H) y temperatura congelados de cada run, elegidos en validación; aquí no se elige ni se ajusta nada con test u OOD.
- F1 macro y ocupado→libre con umbral 0.5; el valor, en el punto de operación de cada escenario. Ocupado→libre = P(mostrado libre | ocupado), el error más caro.
- Con varias semillas: media ± desviación estándar entre semillas, y el IC95 es el promedio de los IC95 bootstrap de cada semilla (igual que `iteraciones.md`).
- Δ pareado: en cada remuestra se usan las mismas imágenes (emparejadas por ruta) para los dos modelos. "Final > ref." es la proporción de remuestras en que el final da más valor; no es un p-valor. Compara un solo run de cada modelo, así que no incluye la variación por la semilla de entrenamiento: una diferencia pequeña (como final − B2 en test) puede depender de la semilla.
- Las imágenes de un mismo frame y día están correlacionadas (misma cámara, luz y clima), y el bootstrap por imagen las trata como independientes, así que los IC95 probablemente son más angostos de lo real, sobre todo en OOD.
