import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
import matplotlib.pyplot as plt

file_path = 'E:\\staj\\customers-500000.csv'
data = pd.read_csv(file_path)
data['Subscription Date'] = pd.to_datetime(data['Subscription Date'])
daily_subscriptions = data.groupby('Subscription Date').size().reset_index(name='y')
daily_subscriptions.rename(columns={'Subscription Date': 'ds'}, inplace=True)

model = Prophet()
model.fit(daily_subscriptions)

periods = {'3_months': 90, '6_months': 180, '1_year': 365}
results = {}

for period_name, horizon in periods.items():
    df_cv = cross_validation(model, initial='365 days', period='90 days', horizon=f'{horizon} days')
    df_p = performance_metrics(df_cv)
    results[period_name] = (df_cv, df_p)
def plot_performance_metrics(results):
    metrics = ['mse', 'rmse', 'mae', 'mape', 'coverage']
    num_periods = len(results)
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_periods, num_metrics, figsize=(20, 12), squeeze=False)

    for i, (period_name, (df_cv, df_p)) in enumerate(results.items()):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            plot_cross_validation_metric(df_cv, metric=metric, ax=ax)
            ax.set_title(f'{metric} for {period_name}')

    plt.tight_layout()
    plt.show()

plot_performance_metrics(results)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

fig = model.plot(forecast)
add_changepoints_to_plot(fig.gca(), model, forecast)
plt.show()

fig2 = model.plot_components(forecast)
plt.show()

for period_name, (df_cv, df_p) in results.items():
    df_cv.to_csv(f'crossvalidation_{period_name}.csv', index=False)
    df_p.to_csv(f'performancemetrics_{period_name}.csv', index=False)

forecast.to_csv('forecastmse.csv', index=False)
