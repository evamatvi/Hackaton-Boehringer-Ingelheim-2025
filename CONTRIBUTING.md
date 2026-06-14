# Guía de Contribución

Ideas de mejora abiertas para el proyecto:

- Conectar la carga de datos directamente al EHR del hospital, en lugar del cargador manual de CSV actual.
- Implementar de forma real la evolución de la función renal (creatinina / eGFR), que ahora mismo es una proyección simulada en `app.py`.
- Mostrar la explicabilidad SHAP por paciente en la interfaz (actualmente los factores determinantes se derivan de reglas fijas).
- Validar el modelo con un dataset externo distinto al UCI CKD, ya que el rendimiento tan alto solo aplica a ese dataset concreto.
- Añadir tests automáticos en `tests/`, especialmente para `data_processor.py` (limpieza y feature engineering).
- Configurar un workflow de CI (GitHub Actions) que ejecute los linters y los tests en cada Pull Request.
- Unificar los nombres de archivo entre el código y la estructura del repositorio.
