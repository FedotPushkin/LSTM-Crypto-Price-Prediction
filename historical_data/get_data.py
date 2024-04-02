from binance import Client
import numpy as np
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression
import sys,os
from xgboost import XGBRegressor
sys.path.append(os.path.join('C:/', 'Users', 'Fedot','Downloads','LSTM-Crypto-Price-Prediction','historical_data'))


def calc_color_prob(method,candles):
    longtails = pd.DataFrame()
    y = []

    for i in range(candles.shape[0] - 1):
        if method(candles.iloc[i]):
            longtails.append(candles.iloc[i])
            y.append(1 if candles.iloc[i + 1]['open'] > candles.iloc[i + 1]['close'] else 0)
    sum = 0
    for i in y:
        sum += i
    print(method.__name__ + " result" + str(float(sum) / float(len(y))))
def isredlongtail(candle):
    h = candle['high']
    l = candle['low']
    o = candle['open']
    c = candle['close']
    if o>c and (h-o)<((c-l)):
        return True
    else: return False
def isredlonghead(candle):
    h = candle['high']
    l = candle['low']
    o = candle['open']
    c = candle['close']
    if o>c and (h-o)>((c-l)):
        return True
    else: return False
def isgreenlonghead(candle):
    h = candle['high']
    l = candle['low']
    o = candle['open']
    c = candle['close']
    if o < c and  (h - c) > (o - l):
        return True
    else:
        return False
def isgreenlongtail(candle):
    h = candle['high']
    l = candle['low']
    o = candle['open']
    c = candle['close']
    if o < c and  (h - c) < (o - l):
        return True
    else:
        return False

def get_data_files(start,end,interval):
    intervals = {240:'4h', 60:'1h',30:'30m', 15:'15m', 5:'5m'}
    #start = '19 Mar, 2024'
    #end = '23 Mar 2024'
    trading_pair = 'BTCUSDT'

    # load key and secret and connect to API
    keys = open('historical_data/binance_key1.txt').readline()
    print('Connecting to Client...')
    api = Client(keys[0], keys[1])

    # fetch desired candles of all data
    print('Fetching data (may take multiple API requests)')
    hist = api.get_historical_klines(trading_pair, intervals[interval], start_str=start,end_str=end)

    print('Finished.')

    # create numpy object with closing prices and volume
    hist = np.array(hist, dtype=np.float32)

    op = hist[:, 1]
    hi = hist[:, 2]
    lo = hist[:, 3]
    cl = hist[:, 4]
    vol =hist[:, 5]
    res = []
    for z in zip(op, cl):
        res.append((z[0] + z[1]) / 2)
    candles = pd.DataFrame({'open':op,'close':cl,'high':hi,'low':lo,'mean':res,'vol':vol})




    #calc_color_prob(isgreenlongtail)
    #calc_color_prob(isgreenlonghead)
    #calc_color_prob(isredlongtail)
    #calc_color_prob(isredlonghead)
    sns.set_theme(style="whitegrid")
    dates = pd.date_range(start=start, periods=len(hist), freq="H")
    candles['Date'] =np.array(dates.to_pydatetime(), dtype=np.datetime64)

    #data = pd.DataFrame(candles, dates)
    #data = data.rolling(7).mean()

    #sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    # data information
    print("\nDatapoints:  {0}".format(hist.shape[0]))
    print("Memory:      {0:.2f} Mb\n".format((hist.nbytes) / 1000000))

    # save to file as numpy object
    np.save("hist_cl", cl)
    np.save("hist_op", op)
    np.save("hist_lo", lo)
    np.save("hist_hi", hi)
    np.save("hist_mean", res)

    return candles

    #model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)


#fig.show()
#plt.show()
