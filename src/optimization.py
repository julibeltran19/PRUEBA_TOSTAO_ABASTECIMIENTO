
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum
from pathlib import Path
from src.config import PROCESSED_DATA

def load_forecast(path: str) -> pd.DataFrame:
    """
    Carga el pronóstico de demanda generado por el modelo.

    Parameters
    ----------
    path : str
        Ruta del archivo CSV con el pronóstico.

    Returns
    -------
    pd.DataFrame
        Pronóstico de demanda.
    """
    forecast = pd.read_csv(
        PROCESSED_DATA / "next_week_forecast.csv"
    )

    forecast["fecha"] = pd.to_datetime(forecast["fecha"])

    return forecast


def load_inventory(path: str) -> pd.DataFrame:
    """
    Carga el inventario actual.

    Parameters
    ----------
    path : str
        Ruta del archivo CSV.

    Returns
    -------
    pd.DataFrame
        Inventario por tienda y producto.
    """

    return pd.read_csv(path)

def load_catalog(path: str) -> pd.DataFrame:
    """
    Carga el catálogo de productos.

    Parameters
    ----------
    path : str
        Ruta del archivo CSV.

    Returns
    -------
    pd.DataFrame
        Información de productos y costos.
    """

    return pd.read_csv(path)

def prepare_optimization_data(
    forecast: pd.DataFrame,
    inventory: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara el dataset que utilizará el modelo de optimización.

    Parameters
    ----------
    forecast : pd.DataFrame
        Pronóstico diario de demanda.

    inventory : pd.DataFrame
        Inventario actual.

    catalog : pd.DataFrame
        Información de costos de los productos.

    Returns
    -------
    pd.DataFrame
        Dataset listo para la optimización.
    """

    forecast = forecast.copy()
    forecast["fecha"] = pd.to_datetime(forecast["fecha"])

    # Demanda total esperada para la siguiente semana (media + rango)
    weekly_demand = (
        forecast
        .groupby(["id_tienda", "id_producto"], as_index=False)
        .agg(
            demanda_semanal=("unidades_vendidas", "sum"),
            demanda_lower=("demanda_lower", "sum"),
            demanda_upper=("demanda_upper", "sum"),
        )
    )

    catalog = catalog[
        [
            "id_producto",
            "costo_unitario",
            "precio_venta",
            "costo_almacenamiento_semanal",
        ]
    ]

    df = weekly_demand.merge(inventory, on=["id_tienda", "id_producto"], how="left")
    df = df.merge(catalog, on="id_producto", how="left")

    return df

def calculate_inventory_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la demanda objetivo por SKU-tienda usando el intervalo
    de confianza del pronóstico (lower/upper) y el trade-off entre
    costo de stockout y costo de overstock (enfoque newsvendor).

    Si el margen perdido por no tener stock (stockout) es mucho mayor
    que el costo de almacenar de más (overstock), el objetivo se acerca
    al límite superior del rango (postura agresiva). Si es al revés,
    se acerca al límite inferior (postura conservadora).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con demanda_lower, demanda_upper, precio_venta,
        costo_unitario y costo_almacenamiento_semanal.

    Returns
    -------
    pd.DataFrame
        Dataset con demanda_objetivo y columnas de diagnóstico.
    """
    df = df.copy()

    # Costo de stockout ≈ margen que se pierde por no vender la unidad
    df["costo_stockout"] = df["precio_venta"] - df["costo_unitario"]
    df["costo_overstock"] = df["costo_almacenamiento_semanal"]

    # Ratio de criticidad (critical fractile del modelo newsvendor)
    df["ratio_criticidad"] = df["costo_stockout"] / (
        df["costo_stockout"] + df["costo_overstock"]
    )

    # Interpolación dentro del intervalo de confianza según el ratio
    df["demanda_objetivo"] = df["demanda_lower"] + df["ratio_criticidad"] * (
        df["demanda_upper"] - df["demanda_lower"]
    )

    # Columna de diagnóstico: cuánto "colchón" se agregó vs. la media
    df["stock_seguridad"] = df["demanda_objetivo"] - df["demanda_semanal"]

    return df

def optimize_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimiza la cantidad a pedir para cada producto y tienda
    minimizando el costo de almacenamiento.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con demanda objetivo, stock actual y costos.

    Returns
    -------
    pd.DataFrame
        Resultado de la optimización con cantidades recomendadas.
    """

    # Crear problema
    problem = LpProblem(
        "Inventory_Optimization",
        LpMinimize
    )

    # Variables de decisión
    pedido = {
        i: LpVariable(
            f"pedido_{i}",
            lowBound=0,
            cat="Integer"
        )
        for i in df.index
    }

    # Función objetivo:
    # minimizar costo de almacenamiento
    problem += lpSum(
        pedido[i] * df.loc[i, "costo_almacenamiento_semanal"]
        for i in df.index
    )

    # Restricciones:
    # stock actual + pedido >= demanda objetivo
    for i in df.index:
        problem += (
            df.loc[i, "stock_actual"]
            + pedido[i]
            >=
            df.loc[i, "demanda_objetivo"]
        )

    # Resolver
    problem.solve()

    # Construir resultados
    result = df.copy()

    result["pedido_recomendado"] = [
        pedido[i].value() for i in df.index
    ]

    result["stock_final_estimado"] = (
        result["stock_actual"]
        + result["pedido_recomendado"]
        - result["demanda_semanal"]
    )

    result["costo_almacenamiento_total"] = (
        result["pedido_recomendado"]
        * result["costo_almacenamiento_semanal"]
    )

    return result

def save_optimization_results(
    results: pd.DataFrame,
    path: str,
) -> None:
    """
    Guarda los resultados de la optimización.

    Parameters
    ----------
    results : pd.DataFrame
        Resultado del modelo de optimización.

    path : str
        Ruta donde se almacenará el archivo CSV.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(path, index=False)