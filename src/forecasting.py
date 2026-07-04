import pandas as pd
import numpy as np
import joblib
from src.features import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
)
from src.training import predict_with_interval

def build_next_day_features(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye las variables necesarias para predecir el siguiente día
    para cada combinación tienda-producto.
    """

    history_df = history_df.copy()

    history_df["fecha"] = pd.to_datetime(history_df["fecha"])

    # Última observación de cada tienda-producto
    future_df = (
        history_df
        .sort_values("fecha")
        .groupby(["id_tienda", "id_producto"])
        .tail(1)
        .copy()
    )

    # Avanzar un día
    future_df["fecha"] = future_df["fecha"] + pd.Timedelta(days=1)
    
    # Variables temporales
    future_df = create_time_features(future_df)
    
    return future_df


def forecast_next_week(history_df, model_path):
    """
    Genera el pronóstico de los próximos 7 días para cada
    combinación tienda-producto utilizando forecasting recursivo.
    """

    # Cargar el modelo
    model = joblib.load(model_path)

    # Copia del histórico para ir agregando predicciones
    history = history_df.copy()

    forecasts = []

    # Aquí irá el forecast recursivo
    for day in range(7):

        # Recalcular lags con el histórico actualizado
        history = update_lag_features(history)

        # Construir el siguiente día
        future_df = build_next_day_features(history)

        # Variables que usa el modelo
        X_future = future_df.drop(
            columns=["unidades_vendidas", "fecha", "nombre"],
            errors="ignore"
        )

        # Predecir demanda
        pred = model.predict(X_future)

        mean_pred, lower, upper = predict_with_interval(
            model,
            X_future,
            confidence=0.90
        )

        future_df["unidades_vendidas"] = mean_pred
        future_df["demanda_lower"] = lower
        future_df["demanda_upper"] = upper

        # Agregar al resultado
        forecasts.append(future_df)

        # Agregar al histórico para la siguiente iteración
        history = pd.concat([history, future_df], ignore_index=True)

    return pd.concat(forecasts, ignore_index=True)

def update_lag_features(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula las variables de rezago y ventanas móviles
    utilizando el histórico actualizado.
    """

    history_df = history_df.sort_values(
        ["id_tienda", "id_producto", "fecha"]
    ).copy()

    grouped = history_df.groupby(
        ["id_tienda", "id_producto"]
    )["unidades_vendidas"]

    #
    history_df = create_lag_features(history_df)
    history_df = create_rolling_features(history_df)

    return history_df

def save_forecast(
    forecast: pd.DataFrame,
    path: str,
) -> None:
    """
    Guarda el pronóstico en un archivo CSV.

    Parameters
    ----------
    forecast : pd.DataFrame
        Pronóstico generado para los próximos días.

    path : str
        Ruta donde se almacenará el archivo.
    """

    forecast.to_csv(path, index=False)