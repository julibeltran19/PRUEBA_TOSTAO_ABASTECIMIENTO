import pandas as pd
from src.config import PROCESSED_DATA


def merge_data(
    ventas: pd.DataFrame,
    catalogo: pd.DataFrame,
    tiendas: pd.DataFrame,
    inventario: pd.DataFrame) -> pd.DataFrame:
    """
    Une la información de ventas con catálogo de productos,
    maestro de tiendas e inventario actual.

    Parameters
    ----------
    ventas : pd.DataFrame
        Histórico de ventas.
    catalogo : pd.DataFrame
        Información de productos.
    tiendas : pd.DataFrame
        Información de tiendas.
    inventario : pd.DataFrame
        Inventario actual por tienda y producto.

    Returns
    -------
    pd.DataFrame
        DataFrame consolidado para el entrenamiento del modelo.
    """

    df = ventas.merge(
        catalogo,
        on="id_producto",
        how="left"
    )

    df = df.merge(
        tiendas,
        on="id_tienda",
        how="left"
    )

    df = df.merge(
        inventario,
        on=["id_tienda", "id_producto"],
        how="left"
    )

    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables temporales a partir de la columna 'fecha'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que contiene la columna 'fecha'.

    Returns
    -------
    pd.DataFrame
        DataFrame con las variables temporales agregadas.
    """

    df = df.copy()

    df["fecha"] = pd.to_datetime(df["fecha"])

    df["day_of_week"] = df["fecha"].dt.dayofweek
    df["week"] = df["fecha"].dt.isocalendar().week.astype(int)
    df["month"] = df["fecha"].dt.month
    df["quarter"] = df["fecha"].dt.quarter
    df["day"] = df["fecha"].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables de rezago (lags) por tienda y producto.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las ventas históricas.

    Returns
    -------
    pd.DataFrame
        DataFrame con las variables lag agregadas.
    """

    df = df.copy()

    group = df.groupby(
        ["id_tienda", "id_producto"]
    )

    df["lag_1"] = group["unidades_vendidas"].shift(1)
    df["lag_7"] = group["unidades_vendidas"].shift(7)
    df["lag_14"] = group["unidades_vendidas"].shift(14)

    return df

def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea estadísticas móviles utilizando únicamente información histórica.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las ventas históricas.

    Returns
    -------
    pd.DataFrame
        DataFrame con las variables rolling agregadas.
    """

    df = df.copy()

    group = df.groupby(
        ["id_tienda", "id_producto"]
    )

    df["rolling_mean_7"] = (
        group["unidades_vendidas"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    df["rolling_std_7"] = (
        group["unidades_vendidas"]
        .shift(1)
        .rolling(window=7)
        .std()
    )

    return df

def clean_training_data(df):
    df = df.dropna().reset_index(drop=True)
    return df


def save_processed_data(df):
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        PROCESSED_DATA / "training_dataset.csv",
        index=False
    )