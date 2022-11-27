import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



dfurl = 'https://raw.githubusercontent.com/herrzilinski/toolbox/main/Datasets/Metro_Interstate_Traffic_Volume.csv'
# df_org = pd.read_csv(dfurl, parse_dates=True, squeeze=True, header=0)
df_org = pd.read_csv('C:\\herrzilinski\\GWU\\DATS 6450 Time Series Analysis & Modeling\\6450_FTP/Metro_Interstate_Traffic_Volume.csv', parse_dates=True, squeeze=True, header=0)

# =============================================
# Data Cleaning
duplicate = sum(df_org.duplicated(subset=['date_time'], keep='last'))
print(f'There are {duplicate} duplicates')
df = df_org.drop_duplicates(subset=['date_time'], keep='last')
df['time'] = pd.to_datetime(df['date_time'], format='%m/%d/%Y %H:%M', infer_datetime_format=True)
# df = df.set_index('date_time')
df = df.set_index('time')
df.drop(columns='date_time', inplace=True)
df.index = pd.DatetimeIndex(df.index)


# =============================================
# Outlier Detection
vQ1 = np.quantile(df['traffic_volume'], 0.25)
vQ3 = np.quantile(df['traffic_volume'], 0.75)
v_IQR = vQ3 - vQ1
print(f'Q1 and Q3 of traffic_volume is {vQ1:.2f} & {vQ3:.2f}')
print(f'IQR for traffic_volume is {v_IQR:.2f}')
print(f'Any traffic_volume < {(vQ1 - 1.5 * v_IQR):.2f} and traffic_volume > {(vQ3 + 1.5 * v_IQR):.2f} is an outlier')

df_clean = df[(df['traffic_volume'] > vQ1 - 1.5 * v_IQR) & (df['traffic_volume'] < vQ3 + 1.5 * v_IQR)]
print(f'There are {df.shape[0] - df_clean.shape[0]} traffic_volume outliers.')


# =============================================
# Subset

df_sub = df_clean['2016-01-01 00:00:00':]
temp = df_sub['traffic_volume']
idx = pd.date_range(start='2016-01-01 00:00:00', end='2018-09-30 23:00', freq='H')
missing = idx.difference(pd.to_datetime(temp.index))
print(f'There are {len(missing)} missing values.')


Y = temp.reindex(idx)
Y.interpolate(inplace=True)


