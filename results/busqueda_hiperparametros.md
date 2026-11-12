# Búsqueda de hiperparámetros (solo validación)

Variantes de mnv3_ft_cost que cambian una sola cosa. Se corrieron con `--runs-dir runs_hp` y semilla 42; la selección usa únicamente val (valor comercial). Los runs no entran a la tabla de iteraciones.

| Run | Cambio frente a mnv3_ft_cost | Val F1 macro | Ocupado→libre | Valor comercial val (Q/mes) | IC95 | Mejor época | Tiempo (s) |
|---|---|---|---|---|---|---|---|
| mnv3_ft_cost | (referencia: lr 3e-4, todo entrenable, 224, con aumento) | 0.9955 | 0.33 % | 53,982 | [53,966; 53,993] | 3 | 773 |
| hp_noaug | sin aumento de datos | 0.9954 | 0.29 % | 53,972 | [53,956; 53,983] | 3 | 469 |
| hp_lr1e4 | lr 1e-4 | 0.9954 | 0.27 % | 53,970 | [53,949; 53,987] | 4 | 855 |
| hp_size128 | entrada 128 en vez de 224 | 0.9958 | 0.16 % | 53,968 | [53,938; 53,992] | 6 | 832 |
| hp_partial | solo los últimos bloques (freeze partial) | 0.9892 | 0.10 % | 53,967 | [53,945; 53,984] | 2 | 607 |
| hp_lr1e3 | lr 1e-3 | 0.9945 | 0.22 % | 53,949 | [53,917; 53,976] | 3 | 739 |

**Conclusión:** ninguna variante supera a mnv3_ft_cost en valor comercial y todas quedan dentro de unos Q30/mes (≈0.06 %) de él, con intervalos que se solapan: el resultado es poco sensible a estos hiperparámetros. Se mantiene la configuración de mnv3_ft_cost para mnv3_ft_seeds.
