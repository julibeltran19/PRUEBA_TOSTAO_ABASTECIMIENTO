# Optimización de Abastecimiento — Caso A

Solución end-to-end para pronóstico de demanda por SKU-Tienda y optimización 
de pedidos, balanceando costo de stockout vs. costo de overstock.

## Estructura del proyecto

```
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                 # Datos originales
│   ├── processed/           # Datasets generados (features, forecasts, resultados)
│   └── external/
│
├── models/
│   └── forecast_model.pkl   # Modelo entrenado (ExtraTreesRegressor)
│
├── notebooks/
│   ├── 1. eda.ipynb
│   ├── 2. feature_engineering.ipynb
│   ├── 3. model.ipynb
│   └── 4. optimizacion.ipynb
│
├── reports/
│   ├── presentation.pdf
│   └── figures/
│
└── src/
    ├── config.py
    ├── preprocessing.py
    ├── features.py
    ├── forecasting.py
    ├── training.py
    ├── evaluation.py
    ├── optimization.py
    └── utils.py
```

## Instalación

> **Nota:** este proyecto se probó con **Python 3.10**. Si tienes varias 
> versiones instaladas, usa `py -3.10` en Windows para asegurar compatibilidad.

```bash
git clone https://github.com/julibeltran19/PRUEBA_TOSTAO_ABASTECIMIENTO.git
cd PRUEBA_TOSTAO_ABASTECIMIENTO
py -3.10 -m venv venv
venv\Scripts\activate      # En Mac/Linux: source venv/bin/activate
py -3.10 -m pip install -r requirements.txt
```

## Ejecución

```bash
py -3.10 main.py
```

Esto corre el pipeline completo: preprocesamiento → entrenamiento/pronóstico → 
optimización de pedido. Los resultados finales quedan en 
`data/processed/optimization_results.csv`.

Para explorar el análisis paso a paso, revisar los notebooks en orden 
(`1. eda.ipynb` → `4. optimizacion.ipynb`).

## Arquitectura de la solución

1. **Preprocesamiento y features** (`preprocessing.py`, `features.py`): 
   limpieza de `ventas_historicas.csv` y construcción de variables predictivas 
   por SKU-Tienda.

2. **Pronóstico de demanda** (`training.py`, `forecasting.py`): se evaluaron 
   4 modelos; el mejor resultó un **ExtraTreesRegressor** (con GridSearch y 
   validación `TimeSeriesSplit` de 5 folds para evitar fuga de información 
   temporal). Se estima además un **intervalo de confianza** calibrado con 
   residuales de validación por SKU-Tienda.

3. **Optimización de pedido** (`optimization.py`): se calcula una demanda 
   objetivo interpolando dentro del intervalo de confianza según el ratio 
   de criticidad entre el costo de stockout (margen perdido) y el costo de 
   overstock (almacenamiento) — enfoque tipo *newsvendor*. La asignación 
   final se resuelve como un problema de programación lineal (PuLP) que 
   minimiza el costo de almacenamiento total sujeto a cubrir esa demanda 
   objetivo.

## Resultados

- R² del modelo de pronóstico: 0.69 (con ~4 meses de historia disponible)
- Cobertura del intervalo de confianza (90%): 90.00% (calibrado con residuales 
  de validación)
- Ratio de criticidad promedio: 0.99 (el modelo prioriza consistentemente evitar 
  stockouts, dado que el margen por unidad supera ampliamente el costo de 
  almacenamiento en este catálogo)
- Costo de almacenamiento total (política optimizada): $485.130
- Margen protegido por el buffer de seguridad: $18.301.842 (ROI ~97x sobre el 
  costo incremental de almacenamiento)

## Limitaciones y próximos pasos

- El histórico disponible (4 meses) limita la capacidad del modelo para 
  capturar estacionalidad completa.
- Próximos pasos: incorporar mayor historia, variables de calendario/festivos 
  y promociones, y monitorear la calibración del intervalo de confianza en 
  producción.
```
