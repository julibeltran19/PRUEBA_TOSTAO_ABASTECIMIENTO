from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)


def evaluate_metrics(y_true, y_pred) -> dict:
    """
    Calcula las métricas de evaluación de un modelo de regresión.

    Parameters
    ----------
    y_true : array-like
        Valores reales.

    y_pred : array-like
        Valores predichos por el modelo.

    Returns
    -------
    dict
        Diccionario con las métricas de evaluación.
    """

    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

    return metrics
#plot_predictions()
#plot_residuals()
#plot_feature_importance()
#save_metrics()