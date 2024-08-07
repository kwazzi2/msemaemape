import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt

#загрузил данные
file_path = 'E:\\staj\\customers-500000.csv'
data = pd.read_csv(file_path)

# преобразовал данные
data['Subscription Date'] = pd.to_datetime(data['Subscription Date'])
daily_subscriptions = data.groupby('Subscription Date').size().reset_index(name='y')

daily_subscriptions.rename(columns={'Subscription Date': 'ds'}, inplace=True)

#обучение модели
model = Prophet()
model.fit(daily_subscriptions)

# периоды
periods = {'3_months': 90, '6_months': 180, '1_year': 365}

results = {}


#сохранение метриков
def save_metrics_to_csv(metrics_df, period_name):
    metrics_df.to_csv(f'performance_metrics_sconsole_{period_name}.csv', index=False)


#кросс-валидация
for period_name, horizon in periods.items():
    df_cv = cross_validation(model, initial='365 days', period='90 days', horizon=f'{horizon} days')
    df_p = performance_metrics(df_cv)
    results[period_name] = df_p

    # сохранение в csv
    save_metrics_to_csv(df_p, period_name)

# вывод метриков
for period_name, metrics in results.items():
    print(f"Метрики производительности для {period_name}:")
    print(metrics[['horizon', 'mse', 'rmse', 'mae', 'mape', 'coverage']].head())

# Создание прогнозов на будущее
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Построение графика прогноза
fig = model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model, forecast)
plt.show()

# Построение компонентов прогноза
fig2 = model.plot_components(forecast)
plt.show()
