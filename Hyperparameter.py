import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import itertools
import numpy as np

# внес и обработал данные
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Subscription Date'] = pd.to_datetime(data['Subscription Date'])
    daily_subscriptions = data.groupby('Subscription Date').size().reset_index(name='y')
    daily_subscriptions.rename(columns={'Subscription Date': 'ds'}, inplace=True)
    return daily_subscriptions

# определение модели
def fit_and_cross_validate(params, daily_subscriptions):
    model = Prophet(**params)
    model.fit(daily_subscriptions)

    # кросс-валидация
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='90 days', parallel="processes")
    df_p = performance_metrics(df_cv)
    return df_p

# настройка гиперпараметров
def tune_hyperparameters(daily_subscriptions):
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }

    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []

    for params in all_params:
        df_p = fit_and_cross_validate(params, daily_subscriptions)
        rmses.append(df_p['rmse'].mean())  # Средний RMSE по всем окнам

    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    print(tuning_results)

    best_params = all_params[np.argmin(rmses)]
    print(f"Best parameters: {best_params}")

    return best_params

# Сохранение
def save_results_to_csv(df_p_best, file_path='metrics from hyperparametr.csv'):
    df_p_best.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")

def main():
    file_path = 'E:\\staj\\customers-500000.csv'
    daily_subscriptions = load_data(file_path)

    best_params = tune_hyperparameters(daily_subscriptions)

    # кросс-валидация с лучшими параметрами
    best_model = Prophet(**best_params)
    best_model.fit(daily_subscriptions)
    df_cv_best = cross_validation(best_model, initial='730 days', period='180 days', horizon='90 days')

    # Оценка производительности
    df_p_best = performance_metrics(df_cv_best)

    # График
    fig = plot_cross_validation_metric(df_cv_best, metric='mape')
    plt.title('MAPE during Cross-Validation')
    plt.show()

    # Сохранение
    save_results_to_csv(df_p_best)

if __name__ == '__main__':
    main()
