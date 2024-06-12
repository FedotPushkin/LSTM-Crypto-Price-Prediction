import numpy as np
import pandas as pd
from macd import Macd
from rsi import StochRsi
from dpo import Dpo
from coppock import Coppock
from ta.volatility import BollingerBands
from ta.trend import PSARIndicator, AroonIndicator, ema_indicator, sma_indicator
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import time
import math
from ta.momentum import AwesomeOscillatorIndicator
# from sklearn.preprocessing import StandardScaler


def build_tt_data(data, params):

    validation_length, validation_lag, timesteps, features = params[:4]
    x_tt = np.empty(shape=(timesteps, 14))
    length = data.shape[0]-validation_length-validation_lag - timesteps

    if length < validation_lag + timesteps:
        raise Exception("no tt data, maybe validation_length too big ")
    sum_time = 0
    for v in range(length):
        t0 = time.time()
        tt_input = data.iloc[v:v + validation_lag + timesteps]

        x_tt_single = np.array(build_tt_piece(data=tt_input, params=params))
        x_tt = np.append(x_tt, x_tt_single, axis=0)
        t1 = time.time()
        curr_lap = t1-t0
        sum_time += curr_lap
        print(f'building train data {v} of {length-1}')
        avg_lap = sum_time/(v+1)
        print(f'expected to last {(length-v)*curr_lap/60:.2f} minutes, avg lap: {avg_lap*100:.2f} ms')
    x_tt = np.delete(x_tt, [range(timesteps)], axis=0)
    return x_tt


def build_tt_piece(data, params):
    # obtain labels
    validation_length, validation_lag, timesteps = params[:3]
    data = data.reset_index()
    lo = data.iloc[:]['low'].astype('float64')
    hi = data.iloc[:]['high'].astype('float64')
    cl = data.iloc[:]['close'].astype('float64')
    vol = data.iloc[:]['vol']
    # op = data.iloc[:]['open'].astype('float64')
    # obtain features
    long = 26
    med = 12
    short = 9
    # zeros = np.zeros(len(lo))
    macd = Macd(cl, med, long, short).values
    ema = ema_indicator(cl, window=long, fillna=False)
    macd_n = macd/ema
    stoch_rsi = StochRsi(cl, period=math.ceil(long/2)).hist_values
    x = list(range(cl.shape[0]))
    grad_rsi = np.gradient(stoch_rsi, x)
    dpo = Dpo(cl, period=math.ceil(short/2)).values/sma_indicator(cl, window=math.ceil(short/2), fillna=False)
    cop = Coppock(cl, wma_pd=math.ceil(long/2), roc_long=med-1, roc_short=short+1).values
    grad_cop = np.gradient(cop, x)
    boll = BollingerBands(close=cl, window=math.ceil(long*0.77))
    boll_h = boll.bollinger_hband()/boll.bollinger_mavg()
    grad_bolh = np.gradient(boll_h, x)
    boll_l = boll.bollinger_lband()/boll.bollinger_mavg()
    grad_boll = np.gradient(boll_l, x)
    psar = PSARIndicator(hi, lo, cl, step=0.02, max_step=0.2, fillna=False)
    sma = sma_indicator(vol, window=long, fillna=False)
    sma_awes = sma_indicator(vol, window=math.ceil(long*1.26), fillna=False)
    obv = OnBalanceVolumeIndicator(cl, vol).on_balance_volume()/sma
    ar = AroonIndicator(hi, lo, window=long)
    aru = ar.aroon_up()
    ard = ar.aroon_down()
    sar_d = psar.psar_down()
    sar_u = psar.psar_up()
    awes = AwesomeOscillatorIndicator(hi, lo, window1=math.ceil(short/2),
                                      window2=math.ceil(long*1.26)).awesome_oscillator()/sma_awes
    # sar_di = psar.psar_down_indicator()
    # sar_ui = psar.psar_up_indicator()
    nan_indices = np.isnan(sar_d)
    for ni in range(len(nan_indices)):
        if nan_indices[ni]:
            sar_d[ni] = sar_u[ni]
        sar_d[ni] = (sar_d[ni]-cl[ni])/cl[ni]
    # truncate bad values and shift label
    xe = np.array([macd_n,
                   stoch_rsi,
                   grad_rsi,
                   sar_d,
                   aru-ard,
                   # ema,
                   # ard,
                   obv,
                   # sar_di,
                   # sar_ui,
                   awes,
                   # sar_u,
                   boll_h,
                   grad_bolh,
                   boll_l,
                   grad_boll,
                   grad_cop,
                   dpo,
                   cop
                   ])

    xe = np.transpose(xe)
    return xe[validation_lag:]


def build_val_data(data, params):
    cl = data[1]
    hi = data[2]
    lo = data[3]
    vol = data[5]
    # op = data[0]
    long = 26
    med = 12
    short = 9
    # zeros = np.zeros(len(lo))
    validation_lag = params[1]
    data = pd.DataFrame(cl).reset_index()[1]
    macd = Macd(data, med, long, short).values
    ema = ema_indicator(data, window=long, fillna=False)
    macd_n = np.array(macd)/np.array(ema)
    stoch_rsi = StochRsi(data, period=math.ceil(long/2)).hist_values
    grad_rsi = np.gradient(stoch_rsi, list(range(data.shape[0])))
    dpo = Dpo(data, period=math.ceil(short/2)).values / sma_indicator(data, window=math.ceil(short/2), fillna=False)
    cop = Coppock(data, wma_pd=math.ceil(long/2), roc_long=med-1, roc_short=short+1).values
    grad_cop = np.gradient(cop, list(range(data.shape[0])))
    boll = BollingerBands(close=data, window=math.ceil(long*0.77))
    boll_h = boll.bollinger_hband()/boll.bollinger_mavg()
    grad_bolh = np.gradient(boll_h, list(range(data.shape[0])))
    boll_l = boll.bollinger_lband()/boll.bollinger_mavg()
    grad_boll = np.gradient(boll_l, list(range(data.shape[0])))
    psar = PSARIndicator(hi, lo, data, step=0.02, max_step=0.2, fillna=False)
    sar_d = psar.psar_down()
    sar_u = psar.psar_up()
    awes = AwesomeOscillatorIndicator(hi, lo, window1=math.ceil(short/2), window2=math.ceil(long * 1.26)).awesome_oscillator()
    sma = np.array(sma_indicator(data, window=long, fillna=False))
    sma_awes = sma_indicator(vol, window=math.ceil(long * 1.26), fillna=False)
    awes_n = np.array(awes)/sma_awes
    nan_indices = np.isnan(sar_d)
    gogo = range(nan_indices.index[0], nan_indices.index[0]+len(nan_indices))
    obv = OnBalanceVolumeIndicator(data, vol).on_balance_volume()
    obv_n = np.array(obv)/sma
    ar = AroonIndicator(hi, lo, window=long)
    aru = ar.aroon_up()
    ard = ar.aroon_down()
    # sar_di = psar.psar_down_indicator()
    # sar_ui = psar.psar_up_indicator()
    for ind in gogo:
        if nan_indices[ind]:
            sar_d[ind] = sar_u[ind]
        sar_d[ind] = (sar_d[ind] - data[ind])/data[ind]
    xv = np.array([macd_n,
                   stoch_rsi,
                   grad_rsi,
                   sar_d,
                   # sar_u,
                   aru - ard,
                   # ard,
                   obv_n,
                   awes_n,
                   boll_h,
                   grad_bolh,
                   boll_l,
                   grad_boll,
                   grad_cop,
                   dpo,
                   cop
                   ])

    xv = np.transpose(xv)
    return xv[validation_lag:]


def shape_data(x_s, training, params):

    timesteps, features = params[2:4]
    # scale data
    if training:
        # scaler = StandardScaler()
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_s = scaler.fit_transform(x_s)

        if not os.path.exists('models'):
            os.mkdir('models')
        joblib.dump(scaler, 'models/scaler.dump')
    else:

        scaler = joblib.load('models/scaler.dump')
        x_s = scaler.fit_transform(x_s)

    # reshape data with timesteps
    reshaped = []
    for t in range(timesteps, x_s.shape[0]+1, timesteps):
        onestep = x_s[t - timesteps:t]
        feat_nums = [range(14)]
        drop_features = [2, 5, 8, 10, 11]
        onestep = onestep[:, np.setdiff1d(feat_nums, drop_features)]
        # onestep = scaler.fit_transform(onestep)
        # onestep = normalise(onestep)
        reshaped.append(onestep)

    # account for data lost in reshaping
    x_s = np.array(reshaped)

    return x_s


def normalise(x_n):
    x_n = x_n.T
    cols_for_norm = {4, 5}
    for vert in range(x_n.shape[0]):
        if vert in cols_for_norm:

            x_n = x_n/x_n[vert][0]
    return x_n.T
