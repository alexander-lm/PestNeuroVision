# PestNeuroVision: Detección Móvil de Plagas mediante YOLO11s

Este repositorio contiene el material complementario para el artículo científico: 
**"Implementación de YOLO11s para la detección de plagas: Visualización de métricas y explicabilidad mediante mapas de calor"**.

## Descripción del Proyecto
PestNeuroVision es un sistema de visión computacional diseñado para la detección automática de plagas. Este repositorio facilita la reproducibilidad del estudio, proporcionando el flujo completo desde el análisis de métricas hasta la exportación para dispositivos móviles.

---

## Estructura del Repositorio

### 1. Notebooks de Análisis (Código Abierto - MIT)
1. **01_Metrics_Visualization.ipynb**: Generación de curvas PR, matrices de confusión y pérdida.
2. **02_Inference_TestSet.ipynb**: Pruebas de detección sobre el set de imágenes.
3. **03_HeatMap_Generation.ipynb**: Visualización de la atención del modelo (explicabilidad).
4. **04_TFLite_Export.ipynb**: Optimización del modelo para integración en Android/iOS.

### 2. Conjunto de Datos (Demo Dataset)
Se incluye un **Mini-dataset de prueba** con imágenes seleccionadas bajo licencias **CC0** y **CC BY 4.0**. Este set permite validar el funcionamiento de los Notebooks sin infringir restricciones de derechos de autor del dataset original.

### 3. Aplicación Móvil (Derechos Reservados)
* **Código Fuente**: Ubicado en `/mobile_app/`. Reservado para fines de registro en **INDECOPI**.
* **Archivo APK**: Disponible en `/mobile_app/release/` para instalación directa y validación rápida por parte de los revisores.

---

## Guía de Inicio Rápido

### En Google Colab:
1. Suba o abra el archivo `.ipynb` deseado en su entorno de Colab.
2. Ejecute las celdas de instalación inicial. El script instalará automáticamente `ultralytics` y descargará los pesos del modelo.

### En Dispositivo Android:
1. Descargue el archivo `.apk` de la carpeta `/mobile_app/release/`.
2. Instale el archivo en su dispositivo y conceda permisos de cámara para la detección en tiempo real.

---

## Licencia y Citación
Este proyecto utiliza un esquema de **licencia mixta**. Consulte `LICENSE.txt` para detalles específicos.

**Citación Sugerida (IEEE):**
[1] J. Pérez, "Test_set_heat_map_PestNeuroVision_2026," PestNeuroVision Project Repository, 2026. [Online]. Disponible: https://doi.org/10.5281/zenodo.0000000

---
**Contacto:** [Tu Correo Electrónico Aquí]