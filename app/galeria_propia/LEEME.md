# Galería local de fotos propias

Esta carpeta guarda fotos de un parqueo propio (por ejemplo, el de la UVG) para analizarlas en
la app. **Su contenido no se versiona** (solo este archivo): las fotos pueden mostrar personas y
placas, y el repositorio es público.

## Formato

- Cada foto: `<nombre>.jpg`, `<nombre>.jpeg` o `<nombre>.png`. El nombre solo puede tener
  letras, números, espacios, `_`, `-` o `.`.
- Sus cajas: `<nombre>.json` al lado, con las coordenadas en píxeles de la foto original
  (ya girada según su orientación EXIF, como se ve en el navegador):

```json
{
  "image": "foto.jpeg",
  "boxes": [
    {"slot_id": "1", "x": 185, "y": 645, "w": 152, "h": 148},
    {"slot_id": "2", "x": 372, "y": 642, "w": 148, "h": 148}
  ]
}
```

`slot_id` es opcional (letras, números, `_` o `-`, máximo 32); si falta, se numera la caja.

## Uso

1. Copia la foto a esta carpeta y recarga la app: aparece en la galería, en la sección
   "Fotos propias" (solo aparece si hay alguna).
2. Si no tiene `.json`, se abre directamente en el modo de dibujo. Dibuja una caja ajustada a
   cada espacio (sin topes de concreto ni árboles) y usa **Guardar cajas**: descarga el
   `.json` con este formato; guárdalo aquí con el mismo nombre que la foto.
3. Con el `.json`, la foto se analiza con esas cajas al elegirla en la galería.

La carpeta se puede cambiar con la variable de entorno `OWN_GALLERY_DIR`.
