# Proyecto de cáncer de mama para practicar MLOps

Ejemplo educativo con Breast Cancer Wisconsin, incluido gratuitamente en scikit-learn. No sirve para diagnóstico médico.

## Ejecutar en Windows

Abre esta carpeta en VS Code, abre una terminal y ejecuta:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python cancer_ligero.py
python -m uvicorn app:app --reload
```

Abre http://127.0.0.1:8000/docs para probar la API. El endpoint `/variables` muestra el orden de las 30 mediciones que requiere `/predecir`. `cancer_ligero.py` guarda el modelo como `cancer_model.pkl`.

Con Docker: `docker build -t cancer-mama .` y después `docker run -p 8000:8000 cancer-mama`.
