import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import Counter
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
# , TimeDistributed
from keras.models import Sequential
from keras.saving import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from keras.utils import to_categorical
from sklearn.metrics import roc_curve
import statistics
import os
import sys
import plotly.express as px
from sklearn.model_selection import StratifiedKFold
# , cross_val_score
from technical_analysis.generate_labels import Genlabels
from technical_analysis.macd import Macd
from technical_analysis.rsi import StochRsi
from technical_analysis.dpo import Dpo
from technical_analysis.coppock import Coppock
from ta.volatility import BollingerBands
from ta.trend import PSARIndicator, AroonIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import AwesomeOscillatorIndicator
from get_data import get_data_files
# from technical_analysis.poly_interpolation import PolyInter

# sys.path.append(os.path.join('C:/', 'Users', 'Fedot', 'Downloads', 'LSTM-Crypto-Price-Prediction', 'historical_data'))
#sys.path.append(os.path.join('historical_data'))
# sys.path.append(os.path.join('technical_analysis'))


def graph(x_g, hold_g, start_g, len_g, col_type):

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
    cand_g = pd.DataFrame(candles).iloc[start_g:start_g+len_g]
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
    trace3 = go.Scatter(x=cand_g[cols[col_type][6]], y=hold_g, name='holding', mode='lines', )
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

    layout = go.Layout(
        title='Labels',
        yaxis=dict(
            title='USDT value'

        ),
        yaxis2=dict(
            title='rsi',
        ),
        yaxis3=dict(
            title='macd',

        )
    )

    # fig2 = go.Figure()#data=data, layout=layout)
    fig2 = make_subplots(rows=5, cols=1)
    fig2.add_trace(trace0, row=1, col=1)
    # fig2.add_trace(trace4, row=1, col=1)
    # fig2.add_trace(trace5, row=1, col=1)
    # fig2.add_trace(trace6, row=3, col=1)
    if trace6:
        fig2.add_trace(trace6, row=3, col=1)
    fig2.add_trace(trace3, row=5, col=1)

    # fig2 = go.Figure(data=trace0)#, layout=layout)
    # py.plot(fig1, filename='../docs/label1.html')
    py.plot(fig2, filename='label2.html')


def plothistories(histories, y_pred_p, yval_p):
    for history in histories:
        # summarize history for accuracy
        plt.figure(1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.figure(2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        fpr_keras, tpr_keras, thresholds_keras = roc_curve(yval_p.T[0], y_pred_p.T[0])

        auc_keras = auc(fpr_keras, tpr_keras)

        plt.figure(3)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        plt.plot(fpr_keras, tpr_keras, label='RF (area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()


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
    obv = OnBalanceVolumeIndicator(cl,vol).on_balance_volume()

    ar = AroonIndicator(hi,lo, window = 25)
    aru = ar.aroon_up()
    ard =  ar.aroon_down()
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
                   #ard,
                   obv,
                   #sar_di,
                   #sar_ui,
                   #awes,
                   # sar_u,
                   #op,
                   #   high,
                   #   low,
                   #cl,
                   #vol,
                  #boll_h,
                   grad_bolh,
                  #boll_l,
                   grad_boll,
                   grad_cop,
                   #dpo,
                   #cop
                   ])
    #   high[30:],
    #   low[30:],
    #   vol[30:],
    xe = np.transpose(xe)
    return xe[validation_lag:]


def build_val_data(data):
    cl = data[1]
    op = data[0]
    hi = data[2]
    lo = data[3]
    vol = data[5]

    data = np.array(cl)
    macd = Macd(data, 6, 12, 3).values
    stoch_rsi = StochRsi(data, period=14).hist_values
    grad_rsi = np.gradient(stoch_rsi, list(range(data.shape[0])))
    dpo = Dpo(data, period=4).values
    cop = Coppock(data, wma_pd=10, roc_long=6, roc_short=3).values
    grad_cop = np.gradient(cop, list(range(data.shape[0])))
    #   inter_slope = PolyInter(data, progress_bar=True).values
    d = pd.DataFrame(data)
    boll = BollingerBands(close=d[0])
    boll_h = boll.bollinger_hband()
    grad_bolh = np.gradient(boll_h, list(range(data.shape[0])))
    boll_l = boll.bollinger_lband()
    grad_boll = np.gradient(boll_l, list(range(data.shape[0])))
    psar = PSARIndicator(hi, lo, cl, step=0.02, max_step=0.2, fillna=False)
    sar_d = psar.psar_down()
    sar_u = psar.psar_up()
    nan_indices = np.isnan(sar_d)
    gogo = range(nan_indices.index[0], nan_indices.index[0]+len(nan_indices))
    obv = OnBalanceVolumeIndicator(cl, vol).on_balance_volume()

    ar = AroonIndicator(hi, lo, window=25)
    aru = ar.aroon_up()
    ard = ar.aroon_down()
    for ind in gogo:
        if nan_indices[ind]:
            sar_d[ind] = sar_u[ind]
        sar_d[ind] = (sar_d[ind] - cl[ind])/cl[ind]
    xv = np.array([macd,
                   stoch_rsi,
                   grad_rsi,
                   sar_d,
                   aru - ard,
                   obv,
                   # sar_u,
                   #op,
                   #   high,
                   #   low,
                   #cl,
                   # vol,
                   #boll_h,
                   grad_bolh,
                   #boll_l,
                   grad_boll,
                   grad_cop,
                   #dpo,
                   #cop
                   ])

    xv = np.transpose(xv)
    return xv[validation_lag:]


def shuffle_and_train(x_adj, y_adj):
    # count the number of each label
    # count_1 = np.count_nonzero(y_adj)
    # count_0 = y_adj.shape[0] - count_1
    # cut = min(count_0, count_1)

    # save some data for testing
    # train_idx = int(cut * split)
    
    # shuffle data
    np.random.seed(42)
    shuffle_index = np.random.permutation(x_adj.shape[0])
    x_adj, y_adj = x_adj[shuffle_index], y_adj[shuffle_index]

    # find indexes of each label
    idx_1 = np.argwhere(y_adj == 1).flatten()
    idx_0 = np.argwhere(y_adj == 0).flatten()

    shuffle_1 = np.random.permutation(len(idx_1))
    shuffle_0 = np.random.permutation(len(idx_0))
    minlen = min(len(idx_1), len(idx_0))
    if len(idx_1) > len(idx_0):
       idx_1 = idx_1[shuffle_0]
    else:
       idx_0 = idx_0[shuffle_1]
    # shuffle_index = np.random.permutation(minlen)

    # grab specified cut of each label put them together
    x_adj = np.concatenate((x_adj[idx_1[:minlen]], x_adj[idx_0[:minlen]]), axis=0)
    # X_test = np.concatenate((x_adj[idx_1[train_idx:cut]], x_adj[idx_0[train_idx:cut]]), axis=0)
    y_adj = np.concatenate((y_adj[idx_1[:minlen]], y_adj[idx_0[:minlen]]), axis=0)
    # y_test = np.concatenate((y_adj[idx_1[train_idx:cut]], y_adj[idx_0[train_idx:cut]]), axis=0)

    # shuffle again to mix labels
    np.random.seed(42)
    shuffle_index = np.random.permutation(x_adj.shape[0])
    x_adj, y_adj = x_adj[shuffle_index], y_adj[shuffle_index]
    bal = Counter(y_adj)
    print(bal.most_common(2))
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for index, (train_indices, val_indices) in enumerate(skf.split(x_adj, y_adj)):
        print("Training on fold " + str(index + 1) + "/5...")
        # Generate batches from indices
        if index == 0:
            xtrain, xval = x_adj[train_indices], x_adj[val_indices]
            ytrain, yval = y_adj[train_indices], y_adj[val_indices]
            ytrain, yval = to_categorical(ytrain, 2), to_categorical(yval, 2)
            # graph(xtrain[0], hold_g=ytrain.T[0], start_g=timesteps, len_g=50, col_type=0)
            model = build_model()
            checkpoint_filepath = 'checkpoint.weights.h5'
            model_checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)
            history = model.fit(xtrain, ytrain, epochs=30, batch_size=32, shuffle=True, validation_data=(xval, yval),
                                callbacks=[model_checkpoint_callback])
            model.load_weights(checkpoint_filepath)
            model.save(f'models/lstm_model{index}.h5')
    y_pred_p = model.predict(xval)
    plothistories([history], y_pred_p, yval)

    #   X_train, y_train = X_train[shuffle_train], y_train[shuffle_train]
    #   X_test, y_test = X_test[shuffle_test], y_test[shuffle_test]


def shape_data(x_s, y_s, training):
    # scale data
    if training:
        scaler = StandardScaler()
    else:
        scaler = joblib.load('models/scaler.dump')
    x_s = scaler.fit_transform(x_s)
    if training:
        if not os.path.exists('models'):
            os.mkdir('models')
        joblib.dump(scaler, 'models/scaler.dump')
    # reshape data with timesteps
    reshaped = []
    for t in range(timesteps, x_s.shape[0]+1):
        onestep = x_s[t - timesteps:t]
        #onestep = scaler.fit_transform(onestep)
        #onestep = normalise(onestep)
        reshaped.append(onestep)

    # account for data lost in reshaping
    x_s = np.array(reshaped)

    y_t = y_s[timesteps:]
    y_t = np.append(y_t, np.random.randint(0, 1+1))
    return x_s, y_t


def normalise(x_n):
    x_n = x_n.T
    cols_for_norm = {4, 5}#, 5, 7}
    for vert in range(x_n.shape[0]):
        if vert in cols_for_norm:

            x_n = x_n/x_n[vert][0]
    return x_n.T


def build_model():
    # first layer
    model = Sequential()
    model.add(LSTM(20, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.15))

    # second layer
    model.add(LSTM(40, return_sequences=False))
    model.add(Dropout(0.15))

    # fourth layer and output
    model.add(Dense(20, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # compile layers
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


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

    for i in range(start_pos, start_pos+len(preds)):
        #   self.savgol = self.apply_filter(deriv=0, hist=candles.iloc[:i + 1]['mean'])
        #   self.savgol_deriv = self.apply_filter(deriv=1, hist=candles.iloc[:i + 1]['mean'])
        #   ####long#####
        if balance > 0 and amount == 0 and position == 0 and preds[i-start_pos] > 0:
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
            if amount > 0 and position == 1 and balance == 0 and preds[i-start_pos] == 0:
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

                if position == 0 and amount == 0 and amount_short == 0 and preds[i-start_pos] < 0:
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
                    if position == - 1 and preds[i-start_pos] > 0 and amount == 0 and amount_short > 0:
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
    y_real = y_val[validation_lag+timesteps:-timesteps]
    guessed_right = 0
    diffs = []
    for g in range(len(preds)):
        if preds[g] == y_real[g][0]:
            guessed_right += 1
        # diffs.append(abs(preds[g]-y_real[g]))
    print(f'guessed right {(100*guessed_right/len(preds)):.2f}%')
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


if __name__ == '__main__':
    start = '20 Jan 2018'
    end = '25 Mar 2024'
    load_data = True
    train = True
    predict = True
    bench = True
    validation_length = 1000
    validation_lag = 30
    timesteps = 15
    if load_data:
        #candles = get_data_files(start, end, 60)
        cl = np.load("hist_cl.npy")
        op = np.load("hist_op.npy")
        lo = np.load("hist_lo.npy")
        hi = np.load("hist_hi.npy")
        res = np.load("hist_mean.npy")
        vol = np.load("vol.npy")
        date = np.load("Date.npy")

        candles = pd.DataFrame({'open': op, 'close': cl, 'high': hi, 'low': lo, 'mean': res, 'vol': vol, 'Date': date})
        X = build_tt_data(candles)
        labels = Genlabels(candles, window=25, polyorder=3).labels
        c = Counter(labels)
        print(c.most_common(2))

        y = labels[31:1 - validation_length]
        #graph(X, start_g=31, hold_g=y, len_g=100, col_type=1)
        y_val = labels[-validation_length:]
        np.save('candles', candles, allow_pickle=True)
        np.save('X', X, allow_pickle=True)
        np.save('y', y, allow_pickle=True)
        np.save('y_val', y_val, allow_pickle=True)
        np.save('val_labels', [0], allow_pickle=True)
    X, y, y_val, candles, val_labels = np.load('X.npy', allow_pickle=True), np.load('y.npy', allow_pickle=True), \
        np.load('y_val.npy', allow_pickle=True), np.load('candles.npy', allow_pickle=True),\
        np.load('val_labels.npy', allow_pickle=True)
    X, y = shape_data(X, y, training=True)
    if train:
        #   ensure equal number of labels, shuffle, and split
        shuffle_and_train(X, y)
    candles = pd.DataFrame(candles)
    #   print(model.summary())
    if predict:
        y_val = to_categorical(y_val, 2)

        y_strat = []
        for models in range(1):
            y_strat.append(list())
            lstm = load_model(f'models/lstm_model{models}.h5')
            for v in range(validation_length - validation_lag - 2*timesteps):
                val_input = candles.iloc[-validation_length+v:-validation_length+v+validation_lag+timesteps]
                X_val = build_val_data(data=val_input)
                # graph(X, start_g=31, hold_g=y, len_g=50, col_type=0)
                X_val, _ = shape_data(X_val, y_val, training=False)

                y_strat[models].append(lstm.predict(X_val)[0][0])
        np.save("y_strat", y_strat, allow_pickle=True)

    y_pred = []
    y_strat = np.load("y_strat.npy",  allow_pickle=True)
    for col in range(len(y_strat[0])):
        vec = y_strat.T[col]
        y_pred.append(statistics.mean(vec))
    # np.save("predictions", y_pred, allow_pickle=True)

    #   preds = list()
    # y_pred = np.load("predictions.npy", allow_pickle=True)
    #   for p in y_pred:
    #   preds.append(p[0])
    #   y_pred=preds
    #   k =y_pred[1]
    #   y_pred_keras = keras_model.predict(X_test).ravel()

    for v in range(len(y_pred)):
        if y_pred[v] < 0.5:
            y_pred[v] = 1
        else:
            if y_pred[v] > 0.5:
                y_pred[v] = 0
            #   else:
            #  y_pred[i] = 0
        #   y_pred = np.array(y_pred)
        #   y_pred = y_pred.transpose()
        #   np.save("predictions", y_pred,allow_pickle=True)
    sh_len, lo_len, mas = list(), list(), list()
    lo_trend, hi_trend = False, False
    counter = 0
    mas.append(y_pred[0])
    #  counting average holding time
    for v in range(1, len(y_pred)):
        if mas[-1] > 0 and y_pred[v] > 0:
            mas[-1] += 1
        elif mas[-1] < 0 and y_pred[v] < 0:
            mas[-1] -= 1
        else:
            mas.append(y_pred[v])

    pos, neg = list(), list()
    for m in mas:
        if m > 0:
            pos.append(m)
        elif m < 0:
            neg.append(m)
    print(f'long {np.mean(pos)} short {np.mean(neg)}')
    #   y_test_trade = y_pred.reshape(y_pred.shape[1],y_pred.shape[0])
    if bench:
        start = candles.shape[0] - validation_length + validation_lag + timesteps
        res = []
        res_s = []
        # random_guess(y_pred)
        _, __, holding_m = strategy_bench(preds=y_pred, start_pos=start, verb=True)
        np.save("holding", holding_m, allow_pickle=True)
        start = len(candles[4]) - 200
        np.save("start", [start], allow_pickle=True)
    else:

        holding_m = np.load("holding.npy")
        start = np.load("start.npy")[0]
    graph(x_g=None, hold_g=holding_m, start_g=start, len_g=199, col_type=0)
