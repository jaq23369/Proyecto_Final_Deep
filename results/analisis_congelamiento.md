# Evaluación única en test y OOD (congelamiento)

Se evaluaron una sola vez todos los runs completos con los umbrales elegidos en val. La selección del mejor modelo (mnv3_ft_cost) se fijó antes, solo con validación.

## Hallazgos

- **Test, condiciones normales:** todos los modelos con deep learning quedan entre Q53,779 y Q53,946 al mes; B1 llega a Q52,176. mnv3_ft_cost es el mejor en test (Q53,946), igual que en val, y la ventaja de mnv3_ft_cost sobre mnv3_ft está dentro del ruido.
- **Lluvia (solo existe en test):** la tasa ocupado→libre de B1 sube a 5.3 % y su F1 baja a 0.955; con MobileNetV3 (mnv3_head a mnv3_ft_cost) esa tasa es de 0.4 % a 0.5 % y el F1 de 0.994 a 0.995.
- **Otro parqueo (OOD, CNRPark A/B):** es donde más pesa el modelo.

  | Modelo | F1 macro | Ocupado→libre | Valor comercial (Q/mes) |
  |---|---|---|---|
  | B1 | 0.840 | 16.1 % | 45,730 |
  | B2 | 0.898 | 2.6 % | 47,927 |
  | malexnet_aug | 0.928 | 2.8 % | 50,282 |
  | mnv3_head | 0.941 | 5.6 % | 49,972 |
  | mnv3_ft | 0.954 | 3.5 % | 52,616 |
  | mnv3_ft_cost | 0.951 | 4.0 % | 52,238 |
  | efficientnet_b0 | 0.957 | 4.9 % | 53,599 |
  | mnv3_ft_seeds (3 semillas) | 0.944 | 3.4 % | 50,999 |

- **Rendimientos decrecientes:** en test, efficientnet_b0 (EfficientNet-B0) no mejora a MobileNetV3; en OOD sí es el de mayor valor (Q53,599 frente a Q52,238 de mnv3_ft_cost), con más parámetros y más cómputo. En OOD la variación entre semillas de mnv3_ft_seeds es grande, así que las diferencias pequeñas entre mnv3_ft, mnv3_ft_cost y mnv3_ft_seeds no son concluyentes.
- **Limitación:** el OOD solo tiene cámaras soleadas y el valor se calcula con los parámetros de las configs en `configs/value/`, todavía marcados como estimados o de literatura.
