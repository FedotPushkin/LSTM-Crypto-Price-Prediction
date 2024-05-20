import pandas as pd
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
validation_lag = 30
timesteps = 15
def graph(x_g, hold_g, start_g,  len_g, col_type,y_pred_p =[],y_pred_n =[], target=[],savgol =[]):


    candles = np.load('candles.npy', allow_pickle=True)
    lag = 200

    def min_g(a, b):
        temp = list()
        for mn in range(len(a)):
            temp.append(min(a[mn], b[mn]))
        return temp

    def max_g(a, b):
        temp = list()
        for mx in range(len(a)):
            temp.append(min(a[mx], b[mx]))
        return temp

    start_pred = candles.shape[0] - start_g
    preds_lag = validation_lag + timesteps
    cand_g = pd.DataFrame(candles).iloc[start_g+lag+preds_lag:start_g+len_g+lag]

    # pr =y_pred_p[-len_g + timesteps:]
    # for i in range(timesteps):
    #  pr.append(None)
    cols = [[0, 1, 2, 3, 4, 5, 6, 7], ['open', 'close', 'high', 'low', 'mean', 'vol', 'Date']]

    # te_g = x_g.T[0]
    # open_g = x_g.T[4]
    # close_g = x_g.T[5]
    bias_x = 1
    if x_g is None:
        trace6 = False
    else:
        trace6 = True

    trace0 = go.Candlestick(x=cand_g[cols[col_type][6]],
                            open=cand_g[cols[col_type][0]],
                            high=cand_g[cols[col_type][2]],
                            low=cand_g[cols[col_type][3]],
                            close=cand_g[cols[col_type][1]],
                            name='candles')

    # trace3 = go.Scatter(x=cand_g[cols[col_type][6]], y=hold_g, name='holding', mode='lines', )
    trace3 = go.Scatter(x=cand_g[cols[col_type][6]], y=y_pred_p[lag:], yaxis='y2',
                        name='preds_p', mode='lines+markers', line_color='#008000')
    # trace4 = go.Scatter(x=cand_g[cols[col_type][6]], y=holding_m[lag:], yaxis='y3',
    #                     name='holding_n', mode='lines+markers', line_color='#f44336' )
    # trace7 = go.Scatter(x=cand_g[cols[col_type][6]], y=savgol_deriv[validation_lag+timesteps:], yaxis='y2',
    #                    name='savg_deriv', mode='lines+markers', line_color='#1f88fe')
    trace8 = go.Scatter(x=cand_g[cols[col_type][6]], y=np.array(target[lag:]), yaxis='y4',
                        name='target', mode='lines+markers', line_color='#044336')
    trace9 = go.Scatter(x=cand_g[cols[col_type][6]], y=savgol[lag+preds_lag:],
                        name='savgol', mode='lines+markers', line_color='#14014e')
    trace10 = go.Scatter(y=y_pred_n, name='pp', mode='lines+markers', line_color='#14014e')
    if trace6:
        trace6 = go.Candlestick(x=cand_g[cols[col_type][6]],
                                # x=x_g.T[13][bias_x:len_g],
                                open=x_g.T[4][bias_x:len_g],
                                high=max_g(x_g.T[4][bias_x:len_g], x_g.T[5][bias_x:len_g]),
                                low=min_g(x_g.T[4][bias_x:len_g], x_g.T[5][bias_x:len_g]),
                                close=x_g.T[5][bias_x:len_g],
                                name='xtrain')
        # trace1 = go.Scatter(x=cand_g[cols[col_type][6]], y=x_g.T[0], name='macd', mode='lines',)

        #   trace1 = go.Scatter(x=candles['Date'], y=self.savgol, name='Filter')
        #   trace2 = go.Scatter(x=candles['Date'], y=self.savgol_deriv, name='Derivative', yaxis='y2')
        #   trace2 = go.Scatter(x=cand_g[cols[col_type][6]], y=x_g.T[1], name='rsi', mode='lines',)

        # trace4 = go.Scatter(x=cand_g[cols[col_type][6]], y=x_g.T[3], name='psar_h', mode='lines', )
        # trace5 = go.Scatter(x=cand_g[cols[col_type][6]], y=x_g.T[4], name='psar_l', mode='lines', )
        # data = [trace0, trace1, trace2, trace3]
    y2 = go.YAxis(overlaying='y', side='right')
    y3 = go.YAxis(overlaying='y', side='right')
    y4 = go.YAxis(overlaying='y', side='right')
    y5 = go.YAxis(overlaying='y', side='right')
    layout = go.Layout(
        title='Labels',

        yaxis2=y2,
        yaxis3=y3,
        yaxis4=y4,
        yaxis5=y5
    )
    graphs = [trace10]  # , trace8,,trace3]
    fig2 = go.Figure(data=graphs, layout=layout)
    # fig2 = make_subplots(rows=3, cols=1)
    # fig2.add_trace(trace0, row=1, col=1)
    # fig2.add_trace(trace4, row=1, col=1)
    # fig2.add_trace(trace5, row=1, col=1)
    # fig2.add_trace(trace6, row=3, col=1)
    if trace6:
        fig2.add_trace(trace6, row=3, col=1)
    # fig2.add_trace(trace3, row=3, col=1)
    # fig2.add_trace(trace4, row=4, col=1)
    # fig2 = go.Figure(data=trace0)#, layout=layout)
    # py.plot(fig1, filename='../docs/label1.html')
    py.plot(fig2, filename='label2.html')
