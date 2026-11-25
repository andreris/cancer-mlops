"""Ejemplo educativo: entrenar y guardar un modelo de cáncer de mama."""
from pathlib import Path
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=data.target_names))
model_path = Path(__file__).parent / 'cancer_model.pkl'
with model_path.open('wb') as file:
    pickle.dump(model, file)
print('Modelo guardado en:', model_path)
