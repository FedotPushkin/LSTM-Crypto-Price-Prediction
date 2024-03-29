import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.models import Sequential
from keras.saving import save_model,load_model
from sklearn.preprocessing import StandardScaler
import joblib
from keras.utils import to_categorical
import json
import os,sys
import plotly.express as px
sys.path.append(os.path.join('C:/', 'Users', 'Fedot','Downloads','LSTM-Crypto-Price-Prediction','historical_data'))
sys.path.append(os.path.join('C:/', 'Users', 'Fedot','Downloads','LSTM-Crypto-Price-Prediction','technical_analysis'))

from get_data import get_data_files

from technical_analysis.generate_labels import Genlabels
from technical_analysis.macd import Macd
from technical_analysis.rsi import StochRsi
from technical_analysis.poly_interpolation import PolyInter
from technical_analysis.dpo import Dpo
from technical_analysis.coppock import Coppock


def graph(candles, holding, start):
    # graph the labels
    # candles=pd.read_pickle('../historical_data/candles.csv')
    maxres = 0
    pars = [0.4, 0.0]
    for i in range(3):
        for j in range(3):
            pass
    candles = candles.iloc[start:]
    # holding = self.strategy_bench(thresl=0.4, thress=0.2 )[1]

    trace0 = go.Candlestick(x=candles['Date'],
                            open=candles['open'],
                            high=candles['high'],
                            low=candles['low'],
                            close=candles['close'])
    #trace1 = go.Scatter(x=candles['Date'], y=self.savgol, name='Filter')
    #trace2 = go.Scatter(x=candles['Date'], y=self.savgol_deriv, name='Derivative', yaxis='y2')
    trace3 = go.Scatter(x=candles['Date'], y=holding, name='holding',yaxis='y2')#, )
    data = [trace0, trace3]#, trace2, trace3]

    layout = go.Layout(
        title='Labels',
        yaxis=dict(
            title='USDT value'
        ),
        yaxis2=dict(
            title='Derivative of Filter',
            overlaying='y',
            side='right'
        )#,
        #yaxis3=dict(
        #    title='holding',
        #    overlaying='y',
         #   side='right'
       # )
    )

    fig2 = go.Figure(data=data, layout=layout)
    # fig2 = go.Figure(data=trace0)#, layout=layout)
    # py.plot(fig1, filename='../docs/label1.html')
    py.plot(fig2, filename='label2.html')


def extract_tt_data(data,validation_length):
    # obtain labels
    labels = Genlabels(data, window=25, polyorder=3,graph=True).labels
    data =data[:-validation_length]
    # obtain features
    macd = Macd(data, 6, 12, 3).values
    stoch_rsi = StochRsi(data, period=14).hist_values
    dpo = Dpo(data, period=4).values
    cop = Coppock(data, wma_pd=10, roc_long=6, roc_short=3).values
    inter_slope = PolyInter(data, progress_bar=True).values

    # truncate bad values and shift label
    X = np.array([macd[30:-validation_length],
             stoch_rsi[30:-validation_length],
           inter_slope[30:-validation_length],
                   dpo[30:-validation_length],
                   cop[30:-validation_length]])

    X = np.transpose(X)



    return X, labels,

def build_val_data(data):
    data =np.array(data)
    macd = Macd(data, 6, 12, 3).values
    stoch_rsi = StochRsi(data, period=14).hist_values
    dpo = Dpo(data, period=4).values
    cop = Coppock(data, wma_pd=10, roc_long=6, roc_short=3).values
    inter_slope = PolyInter(data, progress_bar=True).values

    X_val = np.array([macd,
                 stoch_rsi,
               inter_slope,
                       dpo,
                       cop])

    X_val = np.transpose(X_val)


    return X_val
def adjust_data(X, y, split=0.8):
    # count the number of each label
    count_1 = np.count_nonzero(y)
    count_0 = y.shape[0] - count_1
    cut = min(count_0, count_1)

    # save some data for testing
    train_idx = int(cut * split)
    
    # shuffle data
    np.random.seed(42)
    shuffle_index = np.random.permutation(X.shape[0])
    X, y = X[shuffle_index], y[shuffle_index]

    # find indexes of each label
    idx_1 = np.argwhere(y == 1).flatten()
    idx_0 = np.argwhere(y == 0).flatten()

    # grab specified cut of each label put them together 
    X_train = np.concatenate((X[idx_1[:train_idx]], X[idx_0[:train_idx]]), axis=0)
    X_test = np.concatenate((X[idx_1[train_idx:cut]], X[idx_0[train_idx:cut]]), axis=0)
    y_train = np.concatenate((y[idx_1[:train_idx]], y[idx_0[:train_idx]]), axis=0)
    y_test = np.concatenate((y[idx_1[train_idx:cut]], y[idx_0[train_idx:cut]]), axis=0)

    # shuffle again to mix labels
    np.random.seed(7)
    shuffle_train = np.random.permutation(X_train.shape[0])
    shuffle_test = np.random.permutation(X_test.shape[0])

    X_train, y_train = X_train[shuffle_train], y_train[shuffle_train]
    X_test, y_test = X_test[shuffle_test], y_test[shuffle_test]

    return X_train, X_test, y_train, y_test

def shape_data(X, y, timesteps=10):
    # scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if not os.path.exists('models'):
        os.mkdir('models')

    joblib.dump(scaler, 'models/scaler.dump')

    # reshape data with timesteps
    reshaped = []
    for i in range(timesteps, X.shape[0] + 1):
        reshaped.append(X[i - timesteps:i])
    
    # account for data lost in reshaping
    X = np.array(reshaped)
    y = y[timesteps - 1:]

    return X, y

def build_model():
    # first layer
    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))

    # second layer
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))

    # fourth layer and output
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # compile layers
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
def strategy_bench(y_pred, start, thresl=0.6, thress=0.1,verb=False):

    start_balance=1000
    start_short_balance = 1000
    short_balance=start_short_balance
    balance = start_balance
    profit_long=[]
    leftover=0
    profit_short=[]
    long_value=0
    short_value=0
    amount =0
    amount_short=0
    trades=0
    trades_short=0
    fee=0.001
    position = 0#'long':1,'short':-1,'wait:0
    holding=[]
   # for i in range(self.window):
     #   holding.append(0)
    for i in range(start+1,start+len(y_pred)):#self.window-1,candles.shape[0]-1):
        #self.savgol = self.apply_filter(deriv=0, hist=candles.iloc[:i + 1]['mean'])
        #self.savgol_deriv = self.apply_filter(deriv=1, hist=candles.iloc[:i + 1]['mean'])
        #####long#####
        if balance>0 and amount==0 and position ==0 and y_pred[i-start]>0:# self.savgol_deriv[i]>thresl:
            ##open long
            long_value=start_balance
            amount = start_balance*(1-fee)/candles.iloc[i-1]['open']
            balance = 0
            position = 1
            holding.append(1)
            if verb: print('long at'+str(candles.iloc[i-1]['Date'])+'for '+ str(candles.iloc[i-1]['open']))

        else:
              ##close long
            if  amount>0 and position==1 and y_pred[i-start]<0:#self.savgol_deriv[i]<thress and
                profit_long.append(((1-fee)*amount*candles.iloc[i-1]['open']-long_value)/long_value)
                leftover += (1-fee)*amount*candles.iloc[i-1]['open']-start_balance
                trades+=1
                balance=start_balance
                amount = 0
                position = 0
                holding.append(0)
                if verb: print('short at' + str(candles.iloc[i-1]['Date']) + 'for ' + str(candles.iloc[i-1]['open']))
            #####short#####
            else:
                #open short

                if position==0  and y_pred[i-start]<0:#self.savgol_deriv[i]<0-thress:
                    position=-1
                    amount_short=short_balance/candles.iloc[i-1]['open']
                    short_value = (1 - fee) * amount_short * candles.iloc[i-1]['open']
                    holding.append(-1)
                else: #close short
                    if position==-1 and y_pred[i-start]>0:#
                        position = 0
                        short_balance+=short_value-(1 + fee) * amount_short * candles.iloc[i-1]['open']
                        profit_short.append((short_value-(1 + fee) * amount_short * candles.iloc[i-1]['open'])/short_value)

                        holding.append(0)
                        trades_short+=1

                    else:
                        holding.append(position)
        #self.savgol = self.apply_filter(deriv=0, hist=candles.iloc[:i + 3]['mean'])
        #self.savgol_deriv = self.apply_filter(deriv=1, hist=candles.iloc[:i + 3]['mean'])


    if verb: print('balance',balance,'amount',amount,'trades',trades,'trades_short ',trades_short)
    result = (leftover+amount*candles.iloc[candles.shape[0]-1]['close'])/start_balance
    result_short = (short_balance + amount_short * candles.iloc[candles.shape[0] - 1]['close']) / start_short_balance
    print('threshhold',thresl,'trades long',trades,',trades_short', trades_short)
    #print('result_long %.2f avg  long %.2f result_short %.2f avg  short %.2f'%(result,statistics.mean(profit_long),result_short,statistics.mean(profit_short)))
    trace1 = px.histogram(profit_long,nbins=400)
    #print(profit_long)
    trace2 = px.histogram(profit_short,nbins=400)
    #fig=go.Figure(data=[trace1,trace2])
    prof=0
    for p in profit_long:
        if p>0:
            prof+=1

    print('winrate long ' + str(prof/len(profit_long)))
    prof = 0
    for p in profit_short:
        if p > 0:
            prof += 1

    print('winrate short ' +str(prof / len(profit_short)))
    trace1.show()
    return result, holding
if __name__ == '__main__':
    start = '20 Mar 2022'
    end = '25 Mar 2024'

    #with open('historical_data/hist_data.json') as f:
      #  data = json.load(f)
    load_data = False
    train = False
    predict = True
    bench = True
    # load and reshape data
    validation_length = 1000
    validation_lag = 60
    if load_data:
        candles = get_data_files(start, end, 60)
        X, labels = extract_tt_data(candles['mean'], validation_length=validation_length)
        y = labels[31:1 - validation_length]
        y_val = labels[-validation_length:]
        np.save('candles', candles, allow_pickle=True)
        np.save('X', X, allow_pickle=True)
        np.save('y', y, allow_pickle=True)
        np.save('y_val', y, allow_pickle=True)

    else:
        X, y,y_val,candles = np.load('X.npy',allow_pickle=True),np.load('y.npy',allow_pickle=True),\
            np.load('y_val.npy',allow_pickle=True),np.load('candles.npy',allow_pickle=True)
    X, y = shape_data(X, y, timesteps=10)

    # ensure equal number of labels, shuffle, and split
    X_train, X_test, y_train, y_test = adjust_data(X, y)
    
    # binary encode for softmax function
    y_train, y_test = to_categorical(y_train, 2), to_categorical(y_test, 2)

    # build and train model
    if train:
        model = build_model()
        model.fit(X_train, y_train, epochs=20, batch_size=32, shuffle=True, validation_data=(X_test, y_test))
        model.save('models/lstm_model.h5')
    else:
        model = load_model('models/lstm_model.h5')
    candles = pd.DataFrame(candles)
    if predict:
        for i in range(validation_length - validation_lag - 10):
            X_val = build_val_data(data=candles[-validation_length+i:-validation_length+validation_lag+i][4])

            y_val = to_categorical(y_val, 2)
            X_val, y_val = shape_data(X_val[-validation_lag:], y_val, timesteps=10)
            y_pred=[]

            X_val_s = np.expand_dims(X_val[i], axis=1)
            preds = model.predict(X_val_s)
            if i==0:
                for p in preds:
                    y_pred.append(p)
            else:
                y_pred.append(preds[-1])


        for i in range(len(y_pred)):
            if y_pred[i][0] < 0.45:
                y_pred[i][0] = 1
            else:
                if y_pred[i][0] > 0.55:y_pred[i][0] = -1
                else: y_pred[i][0] =0
        y_pred = np.array(y_pred)
        y_pred = y_pred.transpose()
        np.save("predictions", y_pred,allow_pickle=True)

    else:
        y_pred = np.load("predictions.npy")
    #

    #y_test_trade = y_pred.reshape(y_pred.shape[1],y_pred.shape[0])
    if bench:
        start =candles.shape[0] - validation_length
        print(strategy_bench(y_pred=y_pred[0], start = start)[0])
        holding=strategy_bench(y_pred=y_pred[0], start=start)[1]
        np.save("holding", holding,allow_pickle=True)
        start = len(candles['mean']) - 200
        np.save("start", [start],allow_pickle=True)
    else:

        holding = np.load("holding.npy")
        start = np.load("start.npy")[0]
    graph(candles=candles, holding=holding, start=start)
