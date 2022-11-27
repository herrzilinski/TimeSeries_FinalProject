import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests
import warnings
warnings.filterwarnings('ignore')

my = requests.get('https://raw.githubusercontent.com/herrzilinski/toolbox/main/handy.py')
open('handy.py', 'w').write(my.text)

from handy import ADF_Cal, kpss_test, GPAC_cal, differencing, ACF_PACF_Plot, SARIMA_Estimate
from DATS6450_Data_Cleaning import Y


# =============================================
# Standardization
Yt = (Y - np.mean(Y)) / np.std(Y)
yt_train, yt_test = train_test_split(Yt, test_size=0.05, shuffle=False)


# =============================================
# Differencing
ACF_PACF_Plot(yt_train, 410, 'Original Series')
W1 = differencing(Yt, season=168, order=1)
ACF_PACF_Plot(W1, 410, series_name='Series After Weekly Differencing')
W1D1 = differencing(W1, season=24, order=1)
ACF_PACF_Plot(W1D1, 410, series_name='Series After Weekly & Daily Differencing')
W1D1H1 = differencing(W1D1, order=1)
ACF_PACF_Plot(W1D1H1, 410, series_name='Series After Weekly, Daily & Hourly Differencing')


# =============================================
# ARMA Parameter Estimation
ADF_Cal(W1D1H1)
kpss_test(W1D1H1)
GPAC_cal(W1D1H1, 25, 12, 12, series_name='Differenced Series')

order_x = [[2, 1, 1], [1, 1, 1, 24], [0, 1, 1, 168]]
model_x = SARIMA_Estimate(yt_train, order=order_x)
params = model_x.parameters(debug_info=True)
model_x.confidence_interval()
model_x.residual_whiteness(lags=170)
model_x.result()
ACF_PACF_Plot(model_x.resid, lags=50, series_name='Model residual')
model_x.plot_prediction(start=-170)

order_x1 = [[23, 1, 0], [0, 1, 1, 24], [0, 1, 1, 168]]
model_x1 = SARIMA_Estimate(yt_train, order=order_x1)
params1 = model_x1.parameters(debug_info=True)
model_x1.confidence_interval()
model_x1.residual_whiteness(lags=170)
model_x1.result()
ACF_PACF_Plot(model_x1.resid, lags=50, series_name='Updated Model residual')
model_x1.plot_prediction(start=-170)

# order_x2 = [[23, 1, 1], [0, 1, 1, 24], [0, 1, 1, 168]]
# model_x2 = SARIMA_Estimate(yt_train, order=order_x2)
# params2 = model_x2.parameters(debug_info=True)
# model_x2.confidence_interval()
# model_x2.residual_whiteness(lags=170)
# model_x2.result()
# ACF_PACF_Plot(model_x2.resid, lags=50, series_name='Updated Model residual')
# model_x2.plot_prediction(start=-170)


# =============================================
# Forecast

steps = len(yt_test)
# steps = 410
plt.plot(yt_test[:steps].tolist(), label='Test Data')
plt.plot(model_x.forecast(steps), label=f'{steps} Steps Ahead Forecast')
plt.title(f'Testing Data Compared With {steps}-Step-Ahead Forecast')
plt.xlabel('# of Forecast Steps')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

y_sarima = model_x.forecast(len(yt_test)) * np.std(Y) + np.mean(Y)
y_test = yt_test * np.std(Y) + np.mean(Y)
rmse_sarima = mean_squared_error(y_sarima, yt_test, squared=False)
print(f'The Root Mean Squared Error of SARIMA forecasting is {rmse_sarima:.2f}.')



