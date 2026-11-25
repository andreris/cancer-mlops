"""API educativa para consultar el modelo entrenado."""
from pathlib import Path
import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer

app = FastAPI(title='Ejemplo cáncer de mama', description='Uso educativo, no clínico')
MODEL_PATH = Path(__file__).parent / 'cancer_model.pkl'

class DatosPaciente(BaseModel):
    variables: list[float] = Field(min_length=30, max_length=30)

@app.get('/')
def inicio():
    return {'mensaje': 'API de ejemplo', 'documentacion': '/docs'}

@app.get('/variables')
def variables():
    return {'orden': load_breast_cancer().feature_names.tolist()}

@app.post('/predecir')
def predecir(datos: DatosPaciente):
    if not MODEL_PATH.exists():
        raise HTTPException(503, 'Primero ejecuta python cancer_ligero.py')
    with MODEL_PATH.open('rb') as file:
        model = pickle.load(file)
    prediccion = int(model.predict([datos.variables])[0])
    return {'resultado': 'benigno' if prediccion == 1 else 'maligno', 'uso': 'Solo educativo'}
