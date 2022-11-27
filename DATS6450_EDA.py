import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import os
import requests


my = requests.get('https://raw.githubusercontent.com/herrzilinski/toolbox/main/handy.py')
open('handy.py', 'w').write(my.text)

from handy import Rolling_Mean_Var, ADF_Cal, kpss_test, ACF_PACF_Plot, backward_selection
from DATS6450_Data_Cleaning import df_clean, df_sub, Y


# =============================================
# Overview
ACF_PACF_Plot(Y, lags=50, series_name='Traffic Volume')
ADF_Cal(Y)
kpss_test(Y)
Rolling_Mean_Var(Y, dataname='Traffic Volume')

corr_mat = df_clean.corr()
sns.heatmap(corr_mat, annot=True)
plt.title(f'Correlation Matrix of Original Feature Space')
plt.tight_layout()
plt.show()


# =============================================
# STL Decomposition
res = STL(Y, period=24).fit()
T = res.trend
S = res.seasonal
R = res.resid

str_trend = np.max([0, 1 - np.var(R)/np.var(R + T)])
print(f'The strength of trend for this data set is {str_trend:.3f}')
str_Season = np.max([0, 1 - np.var(R)/np.var(R + S)])
print(f'The strength of seasonality for this data set is {str_Season:.3f}')

STR = pd.DataFrame({'Seasonality': S, 'Trend': T, 'Residual': R})
fig = px.line(STR)
fig.update_layout(
    title='Traffic Volume After STL Decomposition',
    legend_title='Component',
    xaxis_title='Time',
    yaxis_title='Volume'
)
fig.show(renderer='browser')

# =============================================
# PCA
df_reg = pd.get_dummies(df_sub, columns=['weather_main'], drop_first=True)

features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all',
            'weather_main_Clouds', 'weather_main_Drizzle', 'weather_main_Fog',
            'weather_main_Haze', 'weather_main_Mist', 'weather_main_Rain',
            'weather_main_Smoke', 'weather_main_Snow', 'weather_main_Squall',
            'weather_main_Thunderstorm']
target = 'traffic_volume'

X = df_reg[features].values
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X, columns=features)

PCA_Org = PCA(n_components=len(features))
PCA_Org.fit(X)
cumvar = np.cumsum(PCA_Org.explained_variance_ratio_)

plt.plot(np.arange(1, len(cumvar)+1, 1), cumvar)
plt.xticks(np.arange(1, len(cumvar)+1, 1))
plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.title('Cumulative Ratio of Explained Variance of Original Feature Space')
plt.xlabel('Number of Component')
plt.ylabel('Cumulative Ratio of Explained Variance')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

H = X.T @ X
_, d, _ = np.linalg.svd(H)
print(f'The singular values of original feature space are {d}')
cond_num = np.linalg.cond(X)
print(f'The condition number of original feature space is {cond_num:.2f}, suggesting Degree of Co-linearity is limited.')

PCA_T = PCA(n_components=12)
PCA_T.fit(X)
cumvar_T = np.cumsum(PCA_T.explained_variance_ratio_)

plt.plot(np.arange(1, len(cumvar_T)+1, 1), cumvar_T)
plt.xticks(np.arange(1, len(cumvar_T)+1, 1))
plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.title('Cumulative Ratio of Explained Variance of Transformed Feature Space')
plt.xlabel('Number of Component')
plt.ylabel('Cumulative Ratio of Explained Variance')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

Xt = PCA_T.transform(X)
Ht = Xt.T @ Xt
_, dt, _ = np.linalg.svd(Ht)
print(f'The singular values of transformed feature space are {dt}')
cond_num_t = np.linalg.cond(Xt)
print(f'The condition number of transformed feature space is {cond_num:.2f}, suggesting Degree of Co-linearity is limited.')


# =============================================
# Linear Regression
X = df_reg[features]
Y = df_reg[target]
X = sm.add_constant(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=False, test_size=0.05)
y_train[y_train<=0] = 1

full_lm = sm.OLS(y_train, x_train).fit()
print(full_lm.summary())

features.insert(0, 'const')
regressor = features.copy()
lm_best, ft = backward_selection(y_train, x_train, maxp=0.001)
print(lm_best.summary())
ACF_PACF_Plot(lm_best.resid, 50, 'OLS Model Residual')
Rolling_Mean_Var(lm_best.resid, 'OLS Model Residual')
# for x in ['rain_1h', 'snow_1h', 'weather_main_Drizzle', 'weather_main_Rain',
#           'weather_main_Smoke', 'weather_main_Snow', 'weather_main_Squall']:
#     regressor.remove(x)
#
# lm_1 = sm.OLS(y_train, x_train[regressor]).fit()
# print(lm_1.summary())

y_ols = lm_best.predict(x_test[ft])

# fig, ax = plt.subplots()
# fig.suptitle('Training, Testing & Predicion of Prices using OLS')
# ax.plot(y_test, label='Testing')
# ax.plot(y_ols, label='Forecast')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Traffic Volume')
# fig.tight_layout()
# plt.show()

# =============================================
# Base models
# Holt-Winters
y_avg = pd.Series(np.full(len(y_test), np.mean(y_train)), index=y_test.index)

y_nav = pd.Series(np.full(len(y_test), y_train[-1]), index=y_test.index)

y_dft = []
for i in range(1, len(y_test) + 1):
    y_dft.append(y_train[-1] + i * (y_train[-1] - y_train[0]) / (len(y_train) - 1))
y_dft = pd.Series(y_dft, index=y_test.index)

ses_fit = SimpleExpSmoothing(y_train).fit(initial_level=y_train[0], smoothing_level=0.5, optimized=False)
y_ses = ses_fit.forecast(steps=len(y_test))
y_ses.index = y_test.index

hl_fit = ExponentialSmoothing(y_train, trend='mul', damped_trend=True, seasonal=None).fit()
y_hl = hl_fit.forecast(steps=len(y_test))
y_hl.index = y_test.index

hw_fit = ExponentialSmoothing(y_train, trend='mul', damped_trend=True, seasonal='mul', seasonal_periods=24).fit()
y_hw = hw_fit.forecast(steps=len(y_test))
y_hw.index = y_test.index

methods = ['Average', 'Naive', 'Drift', 'SES', 'Holt Linear', 'Holt-Winters', 'OLS']
yhat_dic = {methods[0]: y_avg, methods[1]: y_nav, methods[2]: y_dft, methods[3]: y_ses, methods[4]: y_hl,
            methods[5]: y_hw, methods[6]: y_ols}

# fig = plt.figure()
# plt.plot(y_test, label='Testing Data')
# for i in range(len(methods)):
#     plt.plot(yhat_dic[methods[i]], label=f'{methods[i]} Forecast')
# plt.title('Comparison of Different Base Model Forecasts')
# plt.xlabel('Time')
# plt.ylabel('Traffic Volume')
# plt.legend()
# fig.set_size_inches(12, 9)
# plt.tight_layout()
# plt.show()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=y_test.index,
    y=y_test,
    mode='lines',
    name='Test Data'
))
for i in range(len(methods)):
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=yhat_dic[methods[i]],
        mode='lines',
        name=methods[i]
    ))
fig.update_layout(
    title='Comparison of Different Base Model Forecasts',
    legend_title='Forecast Methods',
    xaxis_title='Time',
    yaxis_title='Traffic Volume'
)
fig.show(renderer='browser')

mse_dic = {}
for i in methods:
    mse = mean_squared_error(y_test, yhat_dic[i], squared=False)
    mse_dic[i] = mse
    print(f'The Root Mean Squared Error of {i} forecasting is {mse:.2f}.')



