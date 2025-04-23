import pandas as pd
import numpy as np
X_train = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/Finarosalina_Random_Forest_/main/data/processed/X_train2.csv')
y_train = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/Finarosalina_Random_Forest_/main/data/processed/y_train2.csv')
X_test = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/Finarosalina_Random_Forest_/main/data/processed/X_test2.csv')
y_test = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/Finarosalina_Random_Forest_/main/data/processed/y_test2.csv')

y_test

y_train = y_train.squeeze()
y_test = y_test.squeeze()
X_train

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Crear el modelo con boosting
model = GradientBoostingClassifier(n_estimators=17, random_state=42)

# Entrenar con los datos de entrenamiento
model.fit(X_train, y_train)

# Predecir sobre los datos de prueba
y_pred = model.predict(X_test)

# Evaluar
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# modelo con XGBoost
model = XGBClassifier(
    n_estimators=17,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Entrenar
model.fit(X_train, y_train)

# Predecir
y_pred = model.predict(X_test)

# Evaluar
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

ada.fit(X_train, y_train)

ada_y_pred_train = ada.predict(X_train)
ada_y_pred_test = ada.predict(X_test)

ada_accuracy_train = accuracy_score(y_train, ada_y_pred_train)
ada_accuracy_test = accuracy_score(y_test, ada_y_pred_test)

print(f"Accuracy (Train): {ada_accuracy_train}")
print(f"Accuracy (Test): {ada_accuracy_test}")

print("Classification Report (Test):")
print(classification_report(y_test, ada_y_pred_test, target_names=["No Diabético", "Diabético"]))




from sklearn.metrics import classification_report

# Asegúrate de tener esto antes
y_pred_test = model.predict(X_test)

# Luego genera el reporte correctamente
print(classification_report(
    y_test,
    y_pred_test,
    target_names=["No Diabético", "Diabético"]
))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

base_tree = DecisionTreeClassifier(max_depth=1, random_state=42)

ada = AdaBoostClassifier(estimator=base_tree, random_state=42)

param_grid = {
    'n_estimators': [17, 50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

# Realizar GridSearchCV
grid_search = GridSearchCV(
    estimator=ada,
    param_grid=param_grid,
    scoring='recall',  # Cambiar a 'recall' para priorizar la detección de diabéticos
    cv=5,
    n_jobs=-1,  # Usar todos los núcleos disponibles
    refit='recall'  # Reajustar el mejor modelo según recall
)


grid_search.fit(X_train, y_train)


print("Mejores parámetros encontrados: ", grid_search.best_params_)
print("Mejor puntuación (Recall): ", grid_search.best_score_)

print("Resultados para cada combinación de parámetros:")
print(grid_search.cv_results_)


ada_best = AdaBoostClassifier(learning_rate=1.0, n_estimators=200, random_state=42)


ada_best.fit(X_train, y_train)

y_pred_train_best = ada_best.predict(X_train)
y_pred_test_best = ada_best.predict(X_test)


accuracy_train_best = accuracy_score(y_train, y_pred_train_best)
accuracy_test_best = accuracy_score(y_test, y_pred_test_best)


print(f"Accuracy (Train) con los mejores parámetros: {accuracy_train_best}")
print(f"Accuracy (Test) con los mejores parámetros: {accuracy_test_best}")

print("Classification Report (Test) con los mejores parámetros:")
print(classification_report(y_test, y_pred_test_best, target_names=["No Diabético", "Diabético"]))

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report


lgb = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

lgb.fit(X_train, y_train)

y_pred_train_lgb = lgb.predict(X_train)
y_pred_test_lgb = lgb.predict(X_test)

accuracy_train_lgb = accuracy_score(y_train, y_pred_train_lgb)
accuracy_test_lgb = accuracy_score(y_test, y_pred_test_lgb)

print(f"Accuracy (Train) con LGBMClassifier: {accuracy_train_lgb}")
print(f"Accuracy (Test) con LGBMClassifier: {accuracy_test_lgb}")

print("Classification Report (Test) con LGBMClassifier:")
print(classification_report(y_test, y_pred_test_lgb, target_names=["No Diabético", "Diabético"]))

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

lgb = LGBMClassifier(random_state=42)


param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],  
    'num_leaves': [31, 50, 70]  
}


grid_search = GridSearchCV(
    estimator=lgb,
    param_grid=param_grid,
    scoring='recall',  # 'recall'para  mejorar la capacidad de predecir la enfermedad (valores asimetricos de predicción)
    cv=5,  
    n_jobs=-1,  
    verbose=1  
)

grid_search.fit(X_train, y_train)

print("Mejores parámetros encontrados: ", grid_search.best_params_)

y_pred_train_lgb = grid_search.best_estimator_.predict(X_train)
y_pred_test_lgb = grid_search.best_estimator_.predict(X_test)

accuracy_train_lgb = accuracy_score(y_train, y_pred_train_lgb)
accuracy_test_lgb = accuracy_score(y_test, y_pred_test_lgb)

print(f"Accuracy (Train) con LGBMClassifier: {accuracy_train_lgb}")
print(f"Accuracy (Test) con LGBMClassifier: {accuracy_test_lgb}")

print("Classification Report (Test) con LGBMClassifier:")
print(classification_report(y_test, y_pred_test_lgb, target_names=["No Diabético", "Diabético"]))

from lightgbm import LGBMClassifier

model = LGBMClassifier(
    learning_rate=0.2,
    max_depth=5,
    n_estimators=50,
    num_leaves=31,
    is_unbalance=True  
)

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

# Modelo base con is_unbalance activado para tratar de compesar los datos
lgb = LGBMClassifier(
    random_state=42,
    is_unbalance=True
)


param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4],
    'num_leaves': [15, 31],
    'reg_alpha': [0, 0.1, 1],   
    'reg_lambda': [0, 0.1, 1]   
}

# Búsqueda con enfoque en el recall (para detectar diabéticos)
grid_search = GridSearchCV(
    estimator=lgb,
    param_grid=param_grid,
    scoring='recall',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Mejores parámetros encontrados: ", grid_search.best_params_)

from sklearn.metrics import accuracy_score, classification_report

y_pred_test = grid_search.best_estimator_.predict(X_test)
y_pred_train = grid_search.best_estimator_.predict(X_train)

print("Accuracy (Train):", accuracy_score(y_train, y_pred_train))
print("Accuracy (Test):", accuracy_score(y_test, y_pred_test))
print("Classification Report (Test):")
print(classification_report(y_test, y_pred_test, target_names=["No Diabético", "Diabético"]))

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

gb_model = GradientBoostingClassifier(n_estimators=17, random_state=42)
gb_model.fit(X_train, y_train)


xgb_model = XGBClassifier(
    n_estimators=17,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)

ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
ada_model.fit(X_train, y_train)

# AdaBoostClassifier (mejorado)
ada_best_model = AdaBoostClassifier(learning_rate=1.0, n_estimators=200, random_state=42)
ada_best_model.fit(X_train, y_train)

lgb_model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
lgb_model.fit(X_train, y_train)

import joblib

gb_model_path = '/workspaces/Finarosalina_BOOSTING_ML_python_template/models/gb_model.joblib'
xgb_model_path = '/workspaces/Finarosalina_BOOSTING_ML_python_template/models/xgb_model.joblib'
ada_model_path = '/workspaces/Finarosalina_BOOSTING_ML_python_template/models/ada_model.joblib'
ada_best_model_path = '/workspaces/Finarosalina_BOOSTING_ML_python_template/models/ada_best_model.joblib'
lgb_model_path = '/workspaces/Finarosalina_BOOSTING_ML_python_template/models/lgb_model.joblib'

joblib.dump(gb_model, gb_model_path)
joblib.dump(xgb_model, xgb_model_path)
joblib.dump(ada_model, ada_model_path)
joblib.dump(ada_best_model, ada_best_model_path)
joblib.dump(lgb_model, lgb_model_path)


X_train.to_csv('/workspaces/Finarosalina_BOOSTING_ML_python_template/data/processed/X_train.csv', index=False)
y_train.to_csv('/workspaces/Finarosalina_BOOSTING_ML_python_template/data/processed/y_train.csv', index=False)
X_test.to_csv('/workspaces/Finarosalina_BOOSTING_ML_python_template/data/processed/X_test.csv', index=False)
y_test.to_csv('/workspaces/Finarosalina_BOOSTING_ML_python_template/data/processed/y_test.csv', index=False)