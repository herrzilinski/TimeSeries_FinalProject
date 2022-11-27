import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.model_selection import train_test_split
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import io
import base64
import statsmodels.api as sm
import requests


my = requests.get('https://raw.githubusercontent.com/herrzilinski/toolbox/main/handy.py')
open('handy.py', 'w').write(my.text)

from handy import GPAC_cal, ACF_PACF_Plot, differencing, SARIMA_Generate, SARIMA_Estimate
from DATS6450_Data_Cleaning import Y

Yt = (Y - np.mean(Y)) / np.std(Y)
sample = SARIMA_Generate([[[0.5, 0.2], 0, [-0.5]]], 0, 1, 10000).samples()
y_train, y_test = train_test_split(sample, test_size=0.05, shuffle=False)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('My_app', external_stylesheets=external_stylesheets)

my_app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Differencing and Order Estimation', children=[
            html.Div([
                html.H3('Differencing'),
                html.H4('Please enter the order of weekly differencing (season=168):'),
                dcc.Input(id='DW', value=0, type='number'),
                html.Br(),

                html.H4('Please enter the order of daily differencing (season=24):'),
                dcc.Input(id='DD', value=0, type='number'),
                html.Br(),

                html.H4('Please enter the order of hourly differencing (non-seasonal):'),
                dcc.Input(id='dH', value=0, type='number'),
                html.Br(),
            ]),

            html.Div([
                html.H3('GPAC Table'),
                html.Img(id='GPAC'),
                html.P('Length of the Table'),
                dcc.Slider(id='Lkj', min=4, max=20, step=1, value=8,
                           tooltip={"placement": "bottom", "always_visible": True})
            ],
                style={'width': '49%', 'display': 'inline-block'}
            ),

            html.Div([
                html.H3('ACF & PACF Plot'),
                html.Img(id='ACF_PACF'),
                html.P('Maximum Lags considered:'),
                dcc.Slider(id='lags', min=24, max=840, step=24, value=24,
                           marks={24: '24', 48: '48', 72: '72', 168: '168', 840: '840'},
                           tooltip={"placement": "bottom", "always_visible": True})
            ],
                style={'width': '49%', 'float': 'right', 'display': 'inline-block'}
            ),

            html.Div([
                html.Br(),
                html.H4('ADF Test'),
                html.Div(id='adf')
            ],
                style={'width': '49%', 'display': 'inline-block'}
            ),

            html.Div([
                html.Br(),
                html.H4('KPSS Test'),
                html.Div(id='kpss')
            ],
                style={'width': '49%', 'float': 'right', 'display': 'inline-block'}
            ),

            html.Div([
                html.Br(),
                html.H3('Plotting'),
                dcc.Graph(id='lineplot'),
                html.P('Plot Window'),
                dcc.RangeSlider(id='time_window', min=0, max=26642, step=24, value=[0, 1000], pushable=168,
                                tooltip={"placement": "bottom", "always_visible": True})
            ])
        ]),

        dcc.Tab(label='Parameter Estimation', children=[
            html.Div([
                html.H4('Please enter the order of weekly model (season=168):'),
                html.Div([
                    'AR: ',
                    dcc.Input(id='Wna', value=1, type='number'),
                    'MA: ',
                    dcc.Input(id='Wnb', value=1, type='number')
                ]),
                html.Br(),
                html.H4('Please enter the order of daily model (season=24):'),
                html.Div([
                    'AR: ',
                    dcc.Input(id='Dna', value=1, type='number'),
                    'MA: ',
                    dcc.Input(id='Dnb', value=1, type='number')
                ]),
                html.Br(),

                html.H4('Please enter the order of hourly model (non-seasonal):'),
                html.Div([
                    'AR: ',
                    dcc.Input(id='Hna', value=4, type='number'),
                    'MA: ',
                    dcc.Input(id='Hnb', value=1, type='number')
                ]),
                html.Br(),
                html.Button('Fit', id='button', n_clicks=0),

                html.Br(),
                html.Div([
                    html.Div(id='param_output', style={'whiteSpace': 'pre-line'})
                ],
                    style={'width': '49%', 'display': 'inline-block'})
            ]),
        ])
    ])
])


@my_app.callback(
    [Output(component_id='GPAC', component_property='src'),
     Output(component_id='ACF_PACF', component_property='src'),
     Output(component_id='adf', component_property='children'),
     Output(component_id='kpss', component_property='children'),
     Output(component_id='lineplot', component_property='figure')
     ],
    [Input(component_id='DW', component_property='value'),
     Input(component_id='DD', component_property='value'),
     Input(component_id='dH', component_property='value'),
     Input(component_id='Lkj', component_property='value'),
     Input(component_id='lags', component_property='value'),
     Input(component_id='time_window', component_property='value')
     ]
)
def update_graphs(DW, DD, dh, Lkj, lags, time_window):
    W1 = differencing(Yt, season=168, order=DW)
    W1D1 = differencing(W1, season=24, order=DD)
    W1D1H1 = differencing(W1D1, order=dh)

    table = GPAC_cal(W1D1H1, lags, Lkj, Lkj, astable=True)
    buf = io.BytesIO()
    plt.figure()
    sns.heatmap(table, annot=True, vmin=-1, vmax=1, cmap='RdBu')
    plt.title(f'GPAC Table of Differenced Series')
    plt.tight_layout()
    plt.savefig(buf, format="png")  # save to the above file object
    plt.close()
    figdata1 = base64.b64encode(buf.getbuffer()).decode("utf8")

    buf = io.BytesIO()
    fig2, axs = plt.subplots(2, 1)
    sm.graphics.tsa.plot_acf(W1D1H1, lags=lags, ax=axs[0])
    sm.graphics.tsa.plot_pacf(W1D1H1, lags=lags, ax=axs[1])
    fig2.suptitle('ACF/PACF Plot of Differenced Series')
    fig2.tight_layout()
    plt.savefig(buf, format="png")  # save to the above file object
    plt.close()
    figdata2 = base64.b64encode(buf.getbuffer()).decode("utf8")

    adfres = adfuller(W1D1H1)
    adfmsg = f'ADF Statistic is {adfres[0]:.4f}, 1% Critical Value is {list(adfres[4].values())[0]:.4f}'

    kpssres = kpss(W1D1H1, regression='c', nlags="auto")
    kpssmsg = f'KPSS Statistic is {kpssres[0]:.4f}, 10% Critical Value is {list(kpssres[3].values())[0]}'

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=Yt[time_window[0]:time_window[1]],
                              mode='lines',
                              name='Original Data'))
    fig3.add_trace(go.Scatter(y=W1D1H1[time_window[0]:time_window[1]],
                              mode='lines',
                              name='Differenced Data'))
    fig3.update_layout(
        xaxis_title='# of Observation',
        yaxis_title='Value')
    return f'data:image/png;base64,{figdata1}', f'data:image/png;base64,{figdata2}', adfmsg, kpssmsg, fig3


@my_app.callback(
    [Output(component_id='param_output', component_property='children'),
     ],
    [Input('button', 'n_clicks')],
    [State(component_id='DW', component_property='value'),
     State(component_id='DD', component_property='value'),
     State(component_id='dH', component_property='value'),
     State(component_id='Wna', component_property='value'),
     State(component_id='Wnb', component_property='value'),
     State(component_id='Dna', component_property='value'),
     State(component_id='Dnb', component_property='value'),
     State(component_id='Hna', component_property='value'),
     State(component_id='Hnb', component_property='value'),
     ]
)
def estimation_res(n_clicks, DW, DD, dH, Wna, Wnb, Dna, Dnb, Hna, Hnb):
    if n_clicks > 0:
        # order = [[Hna, dH, Hnb], [Dna, DD, Dnb, 24], [Wna, DW, Wnb, 168]]
        # model = SARIMA_Estimate(sample, order)
        # param = model.parameters(debug_info=True)
        # res = param.copy()
        # output = ''
        # for odr in order:
        #     if len(odr) == 3:
        #         odr.append(1)
        #     na = odr[0]
        #     nb = odr[2]
        #     s = odr[3]
        #     if s != 1:
        #         for i in range(na):
        #             output = output + f'The estimated AR{i + 1}_L{s * (i + 1)} is {res[i]}' + '\n'
        #         res = res[na:]
        #         for j in range(nb):
        #             output = output + f'The estimated MA{j + 1}_L{s * (j + 1)} is {res[j]}' + '\n'
        #         res = res[nb:]
        #     else:
        #         for i in range(na):
        #             output = output + f'The estimated AR{i + 1} is {res[i]}' + '\n'
        #         res = res[na:]
        #         for j in range(nb):
        #             output = output + f'The estimated MA{j + 1} is {res[j]}' + '\n'
        #         res = res[nb:]
        return str(DW + DD + dH + Wna + Wnb + Dna + Dnb + Hna + Hnb)



my_app.run_server(port=8052,
                  host='0.0.0.0'
                  )



