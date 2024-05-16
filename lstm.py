


import keras.optimizers
import keras.regularizers as reg
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import Counter
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, AlphaDropout
from keras.callbacks import ModelCheckpoint

# from verstack.stratified_continuous_split import scsplit
# TimeDistributed
from keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from keras.utils import to_categorical
from sklearn.metrics import roc_curve

import statistics
import os
# from keras.src.regularizers

import sys
import plotly.express as px

# cross_val_score
from generate_labels import Genlabels
from macd import Macd
from rsi import StochRsi
from dpo import Dpo
from coppock import Coppock
from shuffle import shuffle_and_train
from ta.volatility import BollingerBands
from ta.trend import PSARIndicator, AroonIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import AwesomeOscillatorIndicator
from xgboost import XGBRegressor, XGBClassifier
from get_data import get_data_files
# from technical_analysis.poly_interpolation import PolyInter

# sys.path.append(os.path.join('C:/', 'Users', 'Fedot', 'Downloads', 'LSTM-Crypto-Price-Prediction', 'historical_data'))
# sys.path.append(os.path.join('historical_data'))
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def graph(x_g, hold_g, start_g,  len_g, col_type):
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




def build_tt_data(data):
    # obtain labels
    op = data.iloc[:-validation_length]['open']
    lo = data.iloc[:-validation_length]['low']
    hi = data.iloc[:-validation_length]['high']
    cl = data.iloc[:-validation_length]['close']
    # date = data.iloc[:-validation_length]['Date']
    vol = data.iloc[:-validation_length]['vol']
    # obtain features
    macd = Macd(cl, 6, 12, 3).values
    stoch_rsi = StochRsi(cl, period=14).hist_values
    x = list(range(cl.shape[0]))
    grad_rsi = np.gradient(stoch_rsi, x)
    dpo = Dpo(cl, period=4).values
    cop = Coppock(cl, wma_pd=10, roc_long=6, roc_short=3).values
    # x_grad = list(range(data.shape[0]-validation_length))
    grad_cop = np.gradient(cop, x)
    #   inter_slope = PolyInter(cl, progress_bar=True).values
    # d =pd.DataFrame(cl)['close']
    boll = BollingerBands(close=cl)
    boll_h = boll.bollinger_hband()
    grad_bolh = np.gradient(boll_h, x)
    boll_l = boll.bollinger_lband()
    grad_boll = np.gradient(boll_l, x)
    psar = PSARIndicator(hi, lo, cl, step=0.02, max_step=0.2, fillna=False)
    awes = AwesomeOscillatorIndicator(hi, lo, window1=5, window2=34).awesome_oscillator()
    obv = OnBalanceVolumeIndicator(cl, vol).on_balance_volume()

    ar = AroonIndicator(hi, lo, window=25)
    aru = ar.aroon_up()
    ard = ar.aroon_down()
    sar_d = psar.psar_down()
    sar_u = psar.psar_up()
    sar_di = psar.psar_down_indicator()
    sar_ui = psar.psar_up_indicator()
    nan_indices = np.isnan(sar_d)
    for ni in range(len(nan_indices)):
        if nan_indices[ni]:
            sar_d[ni] = sar_u[ni]
        sar_d[ni] = (sar_d[ni]-cl[ni])/cl[ni]
    # truncate bad values and shift label
    xe = np.array([macd,
                   stoch_rsi,
                   grad_rsi,
                   sar_d,
                   aru-ard,
                   ard,
                   obv,
                   # sar_di,
                   # sar_ui,
                   # awes,
                   # sar_u,
                   # op,
                   # high,
                   # low,
                   # cl,
                   # vol,
                   #boll_h,
                   grad_bolh,
                   #boll_l,
                   grad_boll,
                   grad_cop,
                   dpo,
                   cop
                   ])
    #   high[30:],
    #   low[30:],
    #   vol[30:],
    xe = np.transpose(xe)
    return xe[validation_lag:]


def build_val_data(data):
    cl = data[1]
    # op = data[0]
    hi = data[2]
    lo = data[3]
    # vol = data[5]

    # data = np.array(cl)
    # macd = Macd(data, 6, 12, 3).values
    # stoch_rsi = StochRsi(data, period=14).hist_values
    # grad_rsi = np.gradient(stoch_rsi, list(range(data.shape[0])))
    # dpo = Dpo(data, period=4).values
    # cop = Coppock(data, wma_pd=10, roc_long=6, roc_short=3).values
    # grad_cop = np.gradient(cop, list(range(data.shape[0])))
    # inter_slope = PolyInter(data, progress_bar=True).values
    # d = pd.DataFrame(data)
    # boll = BollingerBands(close=d[0])
    # boll_h = boll.bollinger_hband()
    # grad_bolh = np.gradient(boll_h, list(range(data.shape[0])))
    # boll_l = boll.bollinger_lband()
    # grad_boll = np.gradient(boll_l, list(range(data.shape[0])))
    psar = PSARIndicator(hi, lo, cl, step=0.02, max_step=0.2, fillna=False)
    sar_d = psar.psar_down()
    sar_u = psar.psar_up()
    # sar_di = psar.psar_down_indicator()
    # sar_ui = psar.psar_up_indicator()
    nan_indices = np.isnan(sar_d)
    gogo = range(nan_indices.index[0], nan_indices.index[0]+len(nan_indices))
    # obv = OnBalanceVolumeIndicator(cl, vol).on_balance_volume()

    # ar = AroonIndicator(hi, lo, window=25)
    # aru = ar.aroon_up()
    # ard = ar.aroon_down()
    for ind in gogo:
        if nan_indices[ind]:
            sar_d[ind] = sar_u[ind]
        sar_d[ind] = (sar_d[ind] - cl[ind])/cl[ind]
    xv = np.array([  # macd,
                   # stoch_rsi,
                   # grad_rsi,
                   -sar_d,
                   # sar_u,
                   # aru - ard,
                   # obv,
                   # sar_u,
                   # op,
                   #   high,
                   #   low,
                   # cl,
                   # vol,
                   # boll_h,
                   # grad_bolh,
                   # boll_l,
                   # grad_boll,
                   # grad_cop,
                   # dpo,
                   # cop
                   ])

    xv = np.transpose(xv)
    return xv[validation_lag:]



def shape_data(x_s, y_s, training):
    # scale data
    if training:
        scaler = StandardScaler()
    else:
        pass
        # scaler = joblib.load('models/scaler.dump')
    x_s = scaler.fit_transform(x_s)
    if training:
        if not os.path.exists('models'):
            os.mkdir('models')
        joblib.dump(scaler, 'models/scaler.dump')
    # reshape data with timesteps
    reshaped = []
    for t in range(timesteps, x_s.shape[0]+1):
        onestep = x_s[t - timesteps:t]
        # onestep = scaler.fit_transform(onestep)
        # onestep = normalise(onestep)
        reshaped.append(onestep)

    # account for data lost in reshaping
    x_s = np.array(reshaped)
    if training:
        y_t = y_s[timesteps:]
        y_t = np.append(y_t, np.random.randint(0, 1+1))
    else:
        y_t = y_s
    return x_s, y_t


def normalise(x_n):
    x_n = x_n.T
    cols_for_norm = {4, 5}
    for vert in range(x_n.shape[0]):
        if vert in cols_for_norm:

            x_n = x_n/x_n[vert][0]
    return x_n.T


# @keras.saving.register_keras_serializable()



def strategy_bench(preds, start_pos, verb=False):

    start_balance = 1000
    start_short_balance = 1000
    balance = start_balance
    profit_long = []
    leftover = 0
    short_leftover = 0
    profit_short = []
    long_value = 0
    short_value = 0
    amount = 0
    amount_short = 0
    trades = 0
    trades_short = 0
    fee = 0.001
    # 'long':1,'short':-1,'wait:0
    position = 0
    holding = list()

    for i in range(start_pos, start_pos+len(preds)-1):
        #   self.savgol = self.apply_filter(deriv=0, hist=candles.iloc[:i + 1]['mean'])
        #   self.savgol_deriv = self.apply_filter(deriv=1, hist=candles.iloc[:i + 1]['mean'])
        #   ####long#####
        if preds[i - start_pos] is None:
            continue
        if balance > 0 and amount == 0 and position == 0 and preds[i-start_pos] > 1:
            #   #open long#
            #   money spent on long pos
            long_value = start_balance
            #   amount available after opening long position
            amount = start_balance*(1-fee)/candles.iloc[i][0]
            balance = 0
            position = 1
            holding.append(1)
            if verb:
                print('long at '+str(candles.iloc[i][6])+' for ' + str(candles.iloc[i][0]))

        else:
            # #close long#
            if amount > 0 and position == 1 and balance == 0 and (preds[i-start_pos] < 1):
                profit_long.append(((1-fee)*amount*candles.iloc[i][0]-long_value)/long_value)
                leftover += (1-fee)*amount*candles.iloc[i][0]-start_balance
                trades += 1
                balance = start_balance
                amount = 0
                position = 0
                holding.append(0)
                if verb:
                    print('short at ' + str(candles.iloc[i][6]) + ' for ' + str(candles.iloc[i][0]))
            #   ####short#####
            else:
                #   #open short#

                if position == 0 and amount == 0 and amount_short == 0 and False and preds[i-start_pos] < 0:
                    position = -1
                    # how much was sold
                    amount_short = start_short_balance*(1 - fee)/candles.iloc[i][0]
                    # how much we got for selling
                    short_value = amount_short * candles.iloc[i][0]
                    holding.append(-1)
                    if verb:
                        print('short at ' + str(candles.iloc[i][6]) + ' for ' + str(candles.iloc[i][0]))
                else:
                    #   #close short#
                    if position == - 1 and preds[i-start_pos] > 0 and amount == 0 and amount_short > 0 and False:
                        position = 0
                        curr_profit = short_value-(1 + fee) * amount_short * candles.iloc[i][0]
                        short_leftover += curr_profit
                        profit_short.append(curr_profit/start_short_balance)
                        amount_short = 0
                        # short_balance=start_short_balance
                        holding.append(0)
                        trades_short += 1
                        if verb:
                            print('long at ' + str(candles.iloc[i][6]) + ' for ' + str(candles.iloc[i][0]))
                    else:
                        holding.append(position)
        #   self.savgol = self.apply_filter(deriv=0, hist=candles.iloc[:i + 3]['mean'])
        #   self.savgol_deriv = self.apply_filter(deriv=1, hist=candles.iloc[:i + 3]['mean'])
    if verb:
        print('balance ', balance, ' amount ', amount, ' trades ', trades,
              ' amount short ', amount_short, ' trades_short ', trades_short)
    result = ((balance+leftover+amount*(1-fee)*candles[1][candles.shape[0]-1])/start_balance)-1
    if amount_short > 0:
        short_leftover += start_short_balance - amount_short * (1+fee)*candles[1][candles.shape[0]-1]
    result_short = short_leftover/start_short_balance
    if verb:
        if amount > 0:
            print('liqudating long at  ' + str(candles[2][candles.shape[0]-1]))
        if amount_short > 0:
            print('liqudating short at  ' + str(candles[2][candles.shape[0]-1]))
    y_real = y_val_p[validation_lag+timesteps:-timesteps]
    guessed_right = 0
    diffs = []
    for g in range(len(preds)):
        pass
        # if preds[g] == y_real[g]:
        #   guessed_right += 1
        # diffs.append(abs(preds[g]-y_real[g]))
    # print(f'guessed right {(100*guessed_right/len(preds)):.2f}%')
    #   print('threshhold',thresl,'trades long',trades,',trades_short', trades_short)
    #   print('result_long %.2f avg  long %.2f result_short %.2f avg  short %.2f'%
    #   (result,statistics.mean(profit_long),result_short,statistics.mean(profit_short)))
    trace1 = px.histogram(profit_long, nbins=400, title='longs')
    #   print(profit_long)
    # trace2 = px.histogram(profit_short, nbins=400, title='shorts')
    trace3 = px.histogram(diffs, nbins=50, title='diffs')
    #   fig=go.Figure(data=[trace1,trace2])
    prof = 0
    for p in profit_long:
        if p > 0:
            prof += 1
    if len(profit_long) > 0:
        pass
        #   print('winrate long ' + str(prof/len(profit_long)))
    prof = 0
    for p in profit_short:
        if p > 0:
            prof += 1
    if len(profit_short) > 0:
        pass
        #  print('winrate short ' + str(prof / len(profit_short)))
    trace1.show()
    # trace2.show()
    trace3.show()

    def sign(a):
        if a > 0:
            return '+'
        else:
            return''
    result = f'long {sign(result)} {result*100:.2f}%, short {sign(result_short)}{result_short*100:.2f}%'
    print(result)
    return result, result_short, holding


def random_guess(length):
    # launches strategybench many times on random long/short orders and shows average result
    y_rand = list()
    for _ in range(500):
        avg_hold_dur = 6
        for o in range(0, length, avg_hold_dur):
            n = np.random.random_integers(low=-1, high=1)
            for k in range(avg_hold_dur):
                if o+k < len(y_rand):
                    y_rand[o+k] = n

        a, b, _ = strategy_bench(preds=y_rand, start_pos=start, verb=False)
        res.append(a)
        res_s.append(b)
    print('mean long', np.mean(res), 'mean short', np.mean(res_s))


def bench_cand(pred):
    start_pos = len(candles)-timesteps-100

    profit = list()
    # for j in range(0, 101):
    temp_profit = 0
    for s in range(start_pos, start_pos + len(pred) - 1):
        if pred[s-start_pos] is None:
            pass
        elif pred[s-start_pos]:
            temp_profit += ((1-0.001)*candles[1][s]-(1+0.001)*candles[0][s])/candles[0][s]
    profit.append(temp_profit)
    plt.figure(1)
    plt.plot(profit)
    plt.title('profit ')
    plt.ylabel('sum roi')
    plt.xlabel('pred value(0.3-0.9')
    plt.show()


if __name__ == '__main__':
    start = '01 Jan 2017'
    end = '08 Apr 2024'
    load_data = False
    reshuffle =False
    train = True
    predict = True
    bench = True
    validation_length = 1000
    validation_lag = 30
    timesteps = 15
    if load_data:
        candles = get_data_files(start, end, 60)
        np.save('candles', candles, allow_pickle=True)
        cl = np.load("hist_cl.npy")
        op = np.load("hist_op.npy")
        lo = np.load("hist_lo.npy")
        hi = np.load("hist_hi.npy")
        res = np.load("hist_mean.npy")
        vol = np.load("vol.npy")
        date = np.load("Date.npy")

        candles = pd.DataFrame({'open': op, 'close': cl, 'high': hi, 'low': lo, 'mean': res, 'vol': vol, 'Date': date})
        X = build_tt_data(candles)
        labels_p = Genlabels(candles['close'], window=25, polyorder=3).labels
        cp = Counter(labels_p)
        labels_n = []
        cn = Counter(labels_n)
        #print(f'pos {cp.most_common(2)}')
        #print(f'neg {cn.most_common(2)}')
        y_p = labels_p[31:1 - validation_length]
        y_n = labels_n[31:1 - validation_length]
        # graph(X, start_g=31, hold_g=y, len_g=100, col_type=1)
        y_val_p = labels_p[-validation_length:]
        y_val_n = labels_n[-validation_length:]

        np.save('X', X, allow_pickle=True)
        np.save('y_p', y_p, allow_pickle=True)
        np.save('y_n', y_n, allow_pickle=True)
        np.save('y_val_p', y_val_p, allow_pickle=True)
        np.save('y_val_n', y_val_n, allow_pickle=True)
        # np.save('val_labels', [0], allow_pickle=True)
    X, y_p, y_n, y_val_p, y_val_n, candles = np.load('X.npy', allow_pickle=True), \
        np.load('y_p.npy', allow_pickle=True), \
        np.load('y_n.npy', allow_pickle=True), \
        np.load('y_val_p.npy', allow_pickle=True), \
        np.load('y_val_n.npy', allow_pickle=True), \
        np.load('candles.npy', allow_pickle=True)
    xp = X
    xn = X
    xp, y_p = shape_data(xp, y_p, training=True)
    xn, y_n = shape_data(xn, y_n, training=True)
    if train:
        #   ensure equal number of labels, shuffle, and split
        shuffle_and_train(xp, y_p, 'pos', reshuffle=reshuffle)
        # shuffle_and_train(xn, y_n, 'neg')

    #   print(model.summary())
    candles = pd.DataFrame(candles)
    if predict:

        # y_val_p = to_categorical(y_val_p, 2)
        # y_val_n = to_categorical(y_val_n, 2)

        def predict(tag):

            y_strat_p = list()
            y_strat_p.append([])
            y_strat_n = list()
            y_strat_n.append([])
            # for models in range(1):
            adding = True
            lencount = 1
            up, down = False, False
            # lstm = load_model(f'models/lstm_model_{tag}_{models}.h5')
            # lstm = load_model(f'my_model.keras')
            for v in range(validation_length - validation_lag - 2*timesteps):
                val_input = candles.iloc[-validation_length+v:-validation_length+v+validation_lag+timesteps]
                X_val = build_val_data(data=val_input)
                # graph(X, start_g=31, hold_g=y, len_g=50, col_type=0)
                X_val, _ = shape_data(X_val, [0], training=False)
                # lstm_pred = lstm.predict(X_val, batch_size=1)
                d = X_val[0][14][0]
                print(f'epoch {v} of 700000')
                close = candles.iloc[-validation_length + v + timesteps][1]
                open = candles.iloc[-validation_length + v + timesteps][0]
                re = (close - open) / open
                if d > 0:
                    if up:
                        lencount += 1
                        if len(y_strat_p) <= lencount:
                            y_strat_p.append(list())

                        y_strat_p[lencount].append(re)
                        up = True

                    else:
                        lencount = 0

                        y_strat_p[lencount].append(re)
                        up = True
                        down = False

                elif d < 0:
                    if down:
                        lencount += 1
                        if len(y_strat_n) <= lencount:
                            y_strat_n.append(list())
                        y_strat_n[lencount].append(re)
                        down = True

                    else:
                        lencount = 0
                        y_strat_n[lencount].append(re)
                        up = False
                        down = True
                # u = X_val[0][14][1]
                # y_strat[models].append(u if (np.isnan(d)) else d)
                # lstm_pred[0]
            y_strat_p = pd.DataFrame(y_strat_p)
            y_strat_n = pd.DataFrame(y_strat_n)
            np.save(f'y_strat_{tag}_p', y_strat_p, allow_pickle=True)
            np.save(f'y_strat_{tag}_n', y_strat_n, allow_pickle=True)


        predict('pos')
        # predict('neg')

    def getpreds(tag):
        y_pred = []
        y_strat = np.load(f'y_strat_pos_{tag}.npy',  allow_pickle=True)

        for col in range(y_strat.shape[0]-1):
            column = y_strat[col]
            # for i in range(column.shape[0]-1):
            # if np.isnan(column[i]):
            # column =np.delete(column, i)
            vec_p = np.nanmean(column)

            y_pred.append(vec_p)    # statistics.mean(vec))
            tempsum = 0
            sums = []
            for pr in range(12, len(y_pred)):
                tempsum += y_pred[pr]
                sums.append(tempsum)

        return y_pred

    # np.save("predictions", y_pred, allow_pickle=True)
    #   preds = list()
    # y_pred = np.load("predictions.npy", allow_pickle=True)
    #   for p in y_pred:
    #   preds.append(p[0])
    #   y_pred=preds
    #   k =y_pred[1]
    #   y_pred_keras = keras_model.predict(X_test).ravel()
    def binarypreds(pred):
        if np.isnan(pred):
            return
        for v in range(len(pred)):
            if pred[v] < 0.4:
                pass    # pred[v] = 1
            else:
                if pred[v] > 0.6:
                    pass    # pred[v] = 0
                else:
                    pass    # pred[v] = None

        return pred

    def addnones(pred):
        for _ in range(timesteps + validation_lag):
            pred = np.insert(pred, 0, None, axis=0)
        for _ in range(timesteps):
            pred = np.append(pred, None)
        return pred
    y_pred_p = np.array(getpreds('p'))
    y_pred_n = np.array(getpreds('n'))
    plt.figure(1)
    plt.plot(y_pred_p)
    plt.title('profit ')
    plt.ylabel('sum roi')
    plt.xlabel('pred value(0.3-0.9')
    plt.show()
    plt.figure(2)
    plt.plot(y_pred_n)
    plt.title('profit ')
    plt.ylabel('sum roi')
    plt.xlabel('pred value(0.3-0.9')
    plt.show()
    # y_pred_n = np.array(binarypreds(getpreds('neg')))
    # y_pred_p = y_pred_p - y_pred_n

    def makegrad(pred):
        grad = []
        for pr in range(1, pred.shape[0]):
            fragment = pred[pr - 1:pr + 1]
            frag_grad = np.gradient(fragment, [0, 1])
            grad.append(frag_grad[0])
        return grad


    grad_pred_p = makegrad(y_pred_p)
    # grad_pred_n = makegrad(y_pred_n)
    # y_pred_p = np.delete(y_pred_p, y_pred_p.shape[0] - 1)
    # y_pred_n = np.delete(y_pred_n, y_pred_n.shape[0] - 1)
    # x_pred = list(range(y_pred_p.shape[0]))
    # grad_pred_p_full = np.gradient(y_pred_p, x_pred)
    # grad_pred_n = np.gradient(y_pred_n, x_pred)
    # grad_pred_p = np.insert(grad_pred_p, 0, None, axis=0)
    # grad_pred_n = np.insert(grad_pred_n, 0, None, axis=0)
    hit = 0
    miss = 0

    candles_val = candles[-validation_length:-timesteps]
    savgol = Genlabels(candles_val[1], window=25, polyorder=3).apply_filter(deriv=0, hist=candles_val[1])
    savgol_deriv = Genlabels(candles_val[1], window=25, polyorder=3).apply_filter(deriv=1, hist=candles_val[1])
    for i in range(validation_length - validation_lag - 2 * timesteps-1):
        pass
    # if (y_pred_p[i] > 0.5 and savgol_deriv[i+validation_lag+timesteps]>0.5)
    # or (y_pred_p[i] < 0.5 and savgol_deriv[i +validation_lag+timesteps] < 0.5):
    # hit += 1
    # else:
    #   miss += 1
    #    print(f'savg res {hit / (hit + miss)}')
    target = []
    for i in range(validation_lag+timesteps, validation_length-timesteps):
        # target.append((candles[1][i]-candles[0][i])/candles[0][i])
        # target.append((candles[1][i] > candles[0][i]) )
        target.append(1 if savgol_deriv[i] > 0.0 else 0)
    # x_train_xg = np.array([y_pred_p[:-100], y_pred_n[:-100],
    # np.array(grad_pred_p)[:-100], np.array(grad_pred_n)[:-100]]).T
    # x_test_xg = np.array([y_pred_p[-100:], y_pred_n[-100:],
    # np.array(grad_pred_p)[-100:], np.array(grad_pred_n)[-100:]]).T

    def trainxg(train_data, targets):

        # xtrain, xtest, ytrain, ytest = train_test_split(train_data, targets, test_size=.2)
        # create model instance
        bst = XGBClassifier()
        # fit model
        history = bst.fit(train_data, targets)
        print("Training score: ", bst.score(train_data, targets))
        # make predictions
        return bst
    # y_trainxg = target[1:-100]
    # xg = trainxg(x_train_xg, y_trainxg)
    roi_preds = []  # xg.predict(x_test_xg)
    hit, miss = 0, 0
    for i in range(100):
        pass
        #   if (not roi_preds[i] and not target[-100+i]) or (roi_preds[i] and target[-100+i]) :
        #    hit+=1
        #   else: miss+=1
    # print(f'xg res {hit/(hit+miss)}')
    # y_pred_p =addnones(y_pred_p)
    # y_pred_n = addnones(y_pred_n)
    # grad_pred_p = addnones(grad_pred_p)

    # grad_pred_p = np.delete(grad_pred_p, grad_pred_p.shape[0]-1)
    # grad_pred_n = np.delete(grad_pred_n, grad_pred_n.shape[0] - 1)

    # y_pred_p = np.array(y_pred_p) - np.array(y_pred_n)
    #   else:
    #  y_pred[i] = 0
    #   y_pred = np.array(y_pred)
    #   y_pred = y_pred.transpose()
    #   np.save("predictions", y_pred,allow_pickle=True)
    sh_len, lo_len, mas = list(), list(), list()
    lo_trend, hi_trend = False, False
    counter = 0
    # mas.append(y_pred[0])
    #  counting average holding time
    # for v in range(1, len(y_pred)):
    #    if mas[-1] > 0 and y_pred[v] > 0:
    #       mas[-1] += 1
    #   elif mas[-1] < 0 and y_pred[v] < 0:
    #       mas[-1] -= 1
    #   else:
    #      mas.append(y_pred[v])

    pos, neg = list(), list()
    for m in mas:
        pass    # if m > 0:
        #    pos.append(m)
        # elif m < 0:
        # neg.append(m)

    # print(f'long {np.mean(pos)} short {np.mean(neg)}')
    #   y_test_trade = y_pred.reshape(y_pred.shape[1],y_pred.shape[0])
    if bench:
        # start = candles.shape[0] - validation_length
        res = []
        res_s = []
        # random_guess(y_pred)
        start = candles.shape[0] - validation_length+validation_lag+timesteps
        # _, __, holding_m = strategy_bench(preds=y_pred_p, start_pos=start, verb=True)
        # np.save("holding", holding_m, allow_pickle=True)
        # start = len(candles) - validation_length
        # np.save("start", [start], allow_pickle=True)
    else:
        start = len(candles[6]) - 100
        holding_m = np.load("holding.npy")
        start = np.load("start.npy")[0]
        start = len(candles[6]) - 100
    # bench_cand(y_pred_p)
    # bench_cand(y_pred_n)
    start = candles.shape[0] - validation_length
    graph(x_g=None, hold_g=[], start_g=start, len_g=200, col_type=0)
