from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from src.evaluation import evaluate_metrics
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import joblib
from src.config import PROCESSED_DATA


CATEGORICAL_FEATURES = [
        "id_tienda",
        "id_producto",
        "categoria",
        "ciudad",
    ]

NUMERICAL_FEATURES = [
        "costo_unitario",
        "precio_venta",
        "costo_almacenamiento_semanal",
        "tamaño_m2",
        "stock_actual",
        "day_of_week",
        "week",
        "month",
        "quarter",
        "day",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_std_7",
    ]

def load_training_data() -> pd.DataFrame:
    """
    Carga el dataset procesado para el entrenamiento del modelo.

    Returns
    -------
    pd.DataFrame
        Dataset listo para entrenar el modelo.
    """

    df = pd.read_csv(
        PROCESSED_DATA / "training_dataset.csv"
    )
    
    df["fecha"] = pd.to_datetime(df["fecha"])
 
    return df

def prepare_target(df: pd.DataFrame):
    """
    Separa las variables predictoras (X) y la variable objetivo (y).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de entrenamiento.

    Returns
    -------
    tuple
        X, y
    """

    y = df["unidades_vendidas"]

    X = df.drop(columns=["unidades_vendidas"])

    return X, y

def split_data(
    df: pd.DataFrame,
    forecast_horizon: int = 7
):
    """
    Divide el dataset respetando el orden temporal.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo.

    forecast_horizon : int
        Número de días utilizados para el conjunto de prueba.

    Returns
    -------
    tuple
        train, test
    """

    df = df.copy()

    df["fecha"] = pd.to_datetime(df["fecha"])

    fecha_corte = (
        df["fecha"].max()
        - pd.Timedelta(days=forecast_horizon)
    )

    train = df[df["fecha"] <= fecha_corte]

    test = df[df["fecha"] > fecha_corte]

    return train, test

def prepare_features_target(
    df: pd.DataFrame,
    target: str = "unidades_vendidas",
    drop_columns: list[str] | None = None,
):
    """
    Separa las variables predictoras (X) y la variable objetivo (y).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de entrenamiento o prueba.

    target : str
        Nombre de la variable objetivo.

    drop_columns : list[str]
        Columnas que no deben utilizarse para entrenar el modelo.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X, y
    """

    if drop_columns is None:
        drop_columns = ["fecha", "nombre"]

    X = df.drop(columns=[target] + drop_columns)
    y = df[target]

    return X, y

def build_preprocessor() -> ColumnTransformer:
    """
    Construye el preprocesador utilizado durante el entrenamiento.

    Returns
    -------
    ColumnTransformer
        Pipeline de transformación de variables categóricas y numéricas.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ]
    )

    return preprocessor

def build_models(preprocessor):
    """
    Construye los modelos base que serán comparados.
    """

    models = {

        "Linear Regression":
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]),

        "Random Forest":
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            ))
        ]),

        "Extra Trees":
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", ExtraTreesRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            ))
        ]),

        "Gradient Boosting":
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(
                random_state=42
            ))
        ])

    }

    return models


def compare_models(
    models,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Entrena y compara varios modelos de regresión.

    Returns
    -------
    pd.DataFrame
        Tabla con las métricas de cada modelo.
    """

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        metrics = evaluate_metrics(y_test, pred)

        metrics["Model"] = name

        results.append(metrics)

    results = pd.DataFrame(results)

    return results[
        ["Model", "MAE", "RMSE", "MAPE", "R2"]
    ].sort_values("RMSE")

def optimize_model(
    pipeline,
    X_train,
    y_train,
):
    """
    Optimiza los hiperparámetros del modelo utilizando
    validación cruzada para series temporales.

    Parameters
    ----------
    pipeline
        Pipeline del modelo a optimizar.

    X_train : pd.DataFrame
        Variables de entrenamiento.

    y_train : pd.Series
        Variable objetivo.

    Returns
    -------
    GridSearchCV
        Objeto GridSearchCV entrenado.
    """

    param_grid = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    tscv = TimeSeriesSplit(n_splits=5)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    return grid

def train_best_model(
    model,
    X_train,
    y_train,
):
    model.fit(X_train, y_train)
    return model

def predict(
    model,
    X,
):
    return model.predict(X)

def predict_with_interval(
    model,
    X,
    residuals_val,
    confidence: float = 0.90,
):
    """
    Genera predicción puntual e intervalo de confianza calibrado
    usando los residuales del set de validación/test.

    Parameters
    ----------
    model : Pipeline entrenado
    X : array-like
        Features para predecir.
    residuals_val : array-like
        Residuales (y_real - y_pred) del set de validación, usados
        para calibrar el ancho del intervalo.
    confidence : float
        Nivel de confianza deseado (ej. 0.90 para 90%).

    Returns
    -------
    mean_pred, lower, upper : np.ndarray
    """
    mean_pred = model.predict(X)
    margen = np.quantile(np.abs(residuals_val), confidence)

    lower = mean_pred - margen
    upper = mean_pred + margen

    return mean_pred, lower, upper

def save_model(
    model,
    path,
):
    joblib.dump(model, path)