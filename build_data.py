import numpy as np
import pandas as pd
import tensorflow as tf
from macd import Macd
from rsi import StochRsi
from dpo import Dpo
from coppock import Coppock
from ta.volatility import BollingerBands
from ta.trend import PSARIndicator, AroonIndicator, ema_indicator, sma_indicator, ADXIndicator, CCIIndicator
from ta.volume import OnBalanceVolumeIndicator, sma_ease_of_movement, ChaikinMoneyFlowIndicator
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import time
import math
from ta.momentum import AwesomeOscillatorIndicator
# from sklearn.preprocessing import StandardScaler
import gc

def build_tt_data(data, params):

    validation_length, validation_lag, timesteps, features = params[:4]
    #x_tt = np.empty(shape=(timesteps, 14))
    length = data.shape[0] - validation_length - validation_lag - timesteps

    if length < validation_lag + timesteps:
        raise Exception("no tt data, maybe validation_length too big ")
    sum_time = 0
    x_temp = list()
    for v in range(length):
        if v % 1000 == 0:
            gc.collect()
        t0 = time.time()
        tt_input = data.iloc[v:v + validation_lag + timesteps]

        x_tt_single = build_tt_piece(data=tt_input, params=params)
        #np.append(x_tt, x_tt_single, axis=0)
        x_temp.append(x_tt_single)
        t1 = time.time()
        curr_lap = t1-t0
        sum_time += curr_lap
        print(f'building train data {v} of {length-1}')
        avg_lap = sum_time/(v+1)
        print(f'expected to last {(length-v)*avg_lap/60:.2f} min, '
              f'avg lap: {avg_lap*100:.2f} ms, curr lap {curr_lap*100:.2f} ms')

    #x_tt = np.delete(x_tt, [range(timesteps)], axis=0)
    return np.array(x_temp)


def build_tt_piece(data, params):
    # obtain labels
    validation_length, validation_lag, timesteps = params[:3]
    data = data.reset_index()
    lo, hi, cl, vol = data['low'].astype('float64'), data['high'].astype('float64'), data['close'].astype('float64'), data['vol'].astype('float64')
    # op = data.iloc[:]['open'].astype('float64')
    # obtain features
    long, med, short = 26, 12, 9

    # zeros = np.zeros(len(lo))
    macd = Macd(cl, med, long, short).values
    ema = ema_indicator(cl, window=long, fillna=False)
    macd_n = macd/ema
    stoch_rsi = StochRsi(cl, period=math.ceil(long/2)).hist_values
    x = list(range(cl.shape[0]))
    grad_rsi = np.gradient(stoch_rsi, x)
    dpo = Dpo(cl, period=math.ceil(long*0.77)).values/sma_indicator(cl, window=math.ceil(long*0.77), fillna=False)
    cop = Coppock(cl, wma_pd=math.ceil(long/2)+1, roc_long=med-1, roc_short=short+1).values
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
    aru, ard = ar.aroon_up(),  ar.aroon_down()
    sar_d, sar_u = psar.psar_down(), psar.psar_up()
    awes = AwesomeOscillatorIndicator(hi, lo, window1=math.ceil(short/2),
                                      window2=math.ceil(long*1.26)).awesome_oscillator()/sma_awes
    adx = ADXIndicator(hi, lo, cl, window=math.ceil(long/2))
    adx_p, adx_n, adx_t = adx.adx_pos(), adx.adx_neg(), adx.adx()
    cci = CCIIndicator(hi, lo, cl, window=math.ceil(long/2)).cci()
    #eom = sma_ease_of_movement(hi, lo, vol, window=math.ceil(long/2))
    chai = ChaikinMoneyFlowIndicator(hi, lo, cl, vol, window=math.ceil(long*0.77)).chaikin_money_flow()
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
                   cop,
                   adx_p,
                   adx_n,
                   adx_t,
                   cci,
                   #eom,
                   chai
                   ])

    xe = np.transpose(xe)
    return xe[validation_lag:]


def build_val_data(data, params):
    data = data.reset_index()
    cl, hi, lo, vol = data[1].astype('float64'), data[2].astype('float64'), data[3].astype('float64'), data[5].astype('float64')

    # op = data[0]
    long, med, short = 26, 12, 9
    # zeros = np.zeros(len(lo))
    validation_lag = params[1]
    macd = Macd(cl, med, long, short).values
    ema = ema_indicator(cl, window=long, fillna=False)
    macd_n = np.array(macd)/np.array(ema)
    stoch_rsi = StochRsi(cl, period=math.ceil(long/2)).hist_values
    grad_rsi = np.gradient(stoch_rsi, list(range(cl.shape[0])))
    dpo = Dpo(cl, period=math.ceil(long*0.77)).values / sma_indicator(cl, window=math.ceil(long*0.77), fillna=False)
    cop = Coppock(cl, wma_pd=math.ceil(long/2)+1, roc_long=med-1, roc_short=short+1).values
    grad_cop = np.gradient(cop, list(range(cl.shape[0])))
    boll = BollingerBands(close=cl, window=math.ceil(long*0.77))
    boll_h = boll.bollinger_hband()/boll.bollinger_mavg()
    grad_bolh = np.gradient(boll_h, list(range(cl.shape[0])))
    boll_l = boll.bollinger_lband()/boll.bollinger_mavg()
    grad_boll = np.gradient(boll_l, list(range(cl.shape[0])))
    psar = PSARIndicator(hi, lo, cl, step=0.02, max_step=0.2, fillna=False)
    sar_d, sar_u = psar.psar_down(), psar.psar_up()
    awes = AwesomeOscillatorIndicator(hi, lo,
                                      window1=math.ceil(short/2), window2=math.ceil(long * 1.26)).awesome_oscillator()
    sma = np.array(sma_indicator(cl, window=long, fillna=False))
    sma_awes = sma_indicator(vol, window=math.ceil(long * 1.26), fillna=False)
    awes_n = np.array(awes)/sma_awes
    nan_indices = np.isnan(sar_d)
    gogo = range(nan_indices.index[0], nan_indices.index[0]+len(nan_indices))
    obv = OnBalanceVolumeIndicator(cl, vol).on_balance_volume()
    obv_n = np.array(obv)/sma
    ar = AroonIndicator(hi, lo, window=long)
    aru, ard = ar.aroon_up(), ar.aroon_down()
    adx = ADXIndicator(hi, lo, cl, window=math.ceil(long/2))
    adx_p, adx_n, adx_t = adx.adx_pos(), adx.adx_neg(), adx.adx()
    cci = CCIIndicator(hi, lo, cl, window=math.ceil(long/2)).cci()
    # eom = sma_ease_of_movement(hi, lo, vol, window=math.ceil(long/2))
    chai = ChaikinMoneyFlowIndicator(hi, lo, cl, vol, window=math.ceil(long*0.77)).chaikin_money_flow()
    # sar_di = psar.psar_down_indicator()
    # sar_ui = psar.psar_up_indicator()
    for ind in gogo:
        if nan_indices[ind]:
            sar_d[ind] = sar_u[ind]
        sar_d[ind] = (sar_d[ind] - cl[ind])/cl[ind]
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
                   cop,
                   adx_p,
                   adx_n,
                   adx_t,
                   cci,
                   # eom,
                   chai
                   ])

    xv = np.transpose(xv)
    return xv[validation_lag:]


def shape_data(x_s, training, timesteps, scaler=None):

    # scale data
    if training:
        # scaler = StandardScaler()

        scaler = MinMaxScaler(feature_range=(0, 1))
        x_s = x_s.reshape(x_s.shape[0]*x_s.shape[1], x_s.shape[2])
        if not os.path.exists('models'):
            os.mkdir('models')
        joblib.dump(scaler, 'models/scaler.dump')
    if scaler is None:
        raise Exception('could not load scaler, it is created from training data')
    x_s = scaler.fit_transform(x_s)

    # reshape data with timesteps
    reshaped = []
    max_t = x_s.shape[0]+1
    for t in range(timesteps, max_t, timesteps):
        onestep = x_s[t - timesteps:t]
        feat_nums = [range(19)]
        drop_features = [5, 8, 10, 11]
        onestep = onestep[:, np.setdiff1d(feat_nums, drop_features)]
        # onestep = scaler.fit_transform(onestep)
        # onestep = normalise(onestep)
        reshaped.append(onestep)
        print(f'reshaping {(t-timesteps)*100/max_t:.2f} %')

    return np.array(reshaped)


def normalise(x_n):
    x_n = x_n.T
    cols_for_norm = {4, 5}
    for vert in range(x_n.shape[0]):
        if vert in cols_for_norm:

            x_n = x_n/x_n[vert][0]
    return x_n.T
