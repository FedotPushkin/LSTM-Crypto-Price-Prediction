from binance import Client
import numpy as np
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
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
intervals = {240:'4h', 60:'1h',30:'30m', 15:'15m', 5:'5m'}
start = '20 Mar, 2024'
trading_pair = 'BTCUSDT'

# load key and secret and connect to API
keys = open('binance_key1.txt').readline()
print('Connecting to Client...')
api = Client(keys[0], keys[1])

# fetch desired candles of all data
print('Fetching data (may take multiple API requests)')
hist = api.get_historical_klines(trading_pair, intervals[30], start)

print('Finished.')

# create numpy object with closing prices and volume
hist = np.array(hist, dtype=np.float32)

op = hist[:, 1]
hi = hist[:, 2]
lo = hist[:, 3]
cl = hist[:, 4]

candles = pd.DataFrame({'open':op,'close':cl,'high':hi,'low':lo})

def calc_color_prob(method):
    longtails = pd.DataFrame()
    y = []

    for i in range(candles.shape[0]-1):
        if method(candles.iloc[i]):
            longtails.append(candles.iloc[i])
            y.append(1 if candles.iloc[i+1]['open']>candles.iloc[i+1]['close'] else 0)
    sum=0
    for i in y:
        sum+=i

    print(method.__name__+" result"+str(float(sum)/float(len(y))))
#calc_color_prob(isgreenlongtail)
#calc_color_prob(isgreenlonghead)
#calc_color_prob(isredlongtail)
#calc_color_prob(isredlonghead)
sns.set_theme(style="whitegrid")
dates = pd.date_range("1 1 2024", periods=len(hist), freq="D")
candles['Date'] = dates
data = pd.DataFrame(hist, dates)
data = data.rolling(7).mean()

#sns.lineplot(data=data, palette="tab10", linewidth=2.5)
# data information
print("\nDatapoints:  {0}".format(hist.shape[0]))
print("Memory:      {0:.2f} Mb\n".format((hist.nbytes) / 1000000))

# save to file as numpy object
np.save("hist_cl", cl)
np.save("hist_op", op)
np.save("hist_lo", lo)
np.save("hist_hi", hi)
res=[]
for z in zip(op,cl):
    res.append((z[0]+z[1])/2)

np.save("hist_mean", res)
candles.to_pickle('candles.csv')

model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
fig = go.Figure(data=[go.Candlestick(x=candles['Date'],
                open=candles['open'],
                high=candles['high'],
                low=candles['low'],
                close=candles['close'])])

fig.show()
#plt.show()
