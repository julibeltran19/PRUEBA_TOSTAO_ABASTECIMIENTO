from src.preprocessing import load_data
from src.features import (
    merge_data,
    create_time_features,
    create_lag_features,
    create_rolling_features,
    save_processed_data,
    clean_training_data
)

from src.training import (
    load_training_data,
    prepare_target,
    split_data,
    prepare_features_target,
    build_preprocessor,
    build_models,
    compare_models,
    optimize_model,
    train_best_model,
    predict,
    evaluate_metrics,
    predict_with_interval,
    save_model,
)

from src.forecasting import (
    forecast_next_week,
    save_forecast
)

from src.optimization import (
    load_forecast,
    load_inventory,
    load_catalog,
    prepare_optimization_data,
    calculate_inventory_targets,
    optimize_inventory,
    save_optimization_results,
    
)

def main():

    print("="*60)
    print("1. Cargando datos")
    print("="*60)

    ventas, catalogo, tiendas, inventario = load_data()

    print("OK")

    print("="*60)
    print("2. Feature Engineering")
    print("="*60)

    df = merge_data(
        ventas,
        catalogo,
        tiendas,
        inventario
    )

    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = clean_training_data(df)

    save_processed_data(df)

    print("OK")

    print("="*60)
    print("3. Entrenamiento")
    print("="*60)

    df = load_training_data()

    train, test = split_data(df)

    X_train, y_train = prepare_features_target(train)
    X_test, y_test = prepare_features_target(test)

    preprocessor = build_preprocessor()

    models = build_models(preprocessor)

    results = compare_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(results)

    grid = optimize_model(
        models["Extra Trees"],
        X_train,
        y_train
    )

    best_model = train_best_model(
        grid.best_estimator_,
        X_train,
        y_train
    )

    pred = predict(
        best_model,
        X_test
    )

    metrics = evaluate_metrics(
        y_test,
        pred
    )

    print(metrics)

    save_model(
        best_model,
        "models/forecast_model.pkl"
    )

    # Predicción con intervalo de confianza
    mean_pred, lower, upper = predict_with_interval(
        best_model,
        X_test,
        confidence=0.90
    )

    # Verifica qué tan bien calibrado está el intervalo:
    # ¿qué % de los valores reales cayeron dentro del rango?
    coverage = ((y_test >= lower) & (y_test <= upper)).mean()
    print(f"Cobertura del intervalo al 90%: {coverage:.2%}")

    print("OK")

    print("="*60)
    print("4. Forecast")
    print("="*60)
    
    future_forecast = forecast_next_week(
        history_df=df,
        model_path="models/forecast_model.pkl"
    )

    save_forecast(
        future_forecast,
        "data/processed/next_week_forecast.csv"
    )

    print("OK")

    print("="*60)
    print("5. Optimización")
    print("="*60)

    forecast = load_forecast(
        "data/processed/next_week_forecast.csv"
    )

    inventory = load_inventory(
        "data/raw/inventario_actual.csv"
    )

    catalog = load_catalog(
        "data/raw/catalogo_productos.csv"
    )

    df = prepare_optimization_data(
        forecast,
        inventory,
        catalog
    )
    print("OK OPTIMIZACION1")

    optimization_df = calculate_inventory_targets(df)
    
    print("OK OPTIMIZACION2")
    results = optimize_inventory(
        optimization_df
    )

    save_optimization_results(
        results,
        "data/processed/optimization_results.csv"
    )

    print("Proceso terminado correctamente.")

if __name__ == "__main__":
    main()