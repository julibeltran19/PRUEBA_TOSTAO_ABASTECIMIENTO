from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"

def load_data():

    df_ventas = pd.read_csv(DATA_RAW / "ventas_historicas.csv")
    df_catalogo = pd.read_csv(DATA_RAW / "catalogo_productos.csv")
    df_tiendas = pd.read_csv(DATA_RAW / "maestro_tiendas.csv")
    df_inventario = pd.read_csv(DATA_RAW / "inventario_actual.csv")

    return df_ventas, df_catalogo, df_tiendas, df_inventario
    