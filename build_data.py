import numpy as np
import pandas as pd
from macd import Macd
from rsi import StochRsi
from dpo import Dpo
from coppock import Coppock
from ta.volatility import BollingerBands
from ta.trend import PSARIndicator, AroonIndicator
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import StandardScaler
import joblib
import os
# from ta.momentum import AwesomeOscillatorIndicator
validation_lag = 30
timesteps = 15


def build_tt_data(data, validation_length):
    # obtain labels

    lo = data.iloc[:-validation_length]['low']
    hi = data.iloc[:-validation_length]['high']
    cl = data.iloc[:-validation_length]['close']
    vol = data.iloc[:-validation_length]['vol']
    # op = data.iloc[:-validation_length]['open']
    # date = data.iloc[:-validation_length]['Date']
    # obtain features
    macd = Macd(cl, 6, 12, 3).values
    stoch_rsi = StochRsi(cl, period=14).hist_values
    x = list(range(cl.shape[0]))
    grad_rsi = np.gradient(stoch_rsi, x)
    dpo = Dpo(cl, period=4).values
    cop = Coppock(cl, wma_pd=10, roc_long=6, roc_short=3).values
    grad_cop = np.gradient(cop, x)
    boll = BollingerBands(close=cl)
    boll_h = boll.bollinger_hband()
    grad_bolh = np.gradient(boll_h, x)
    boll_l = boll.bollinger_lband()
    grad_boll = np.gradient(boll_l, x)
    psar = PSARIndicator(hi, lo, cl, step=0.02, max_step=0.2, fillna=False)
    obv = OnBalanceVolumeIndicator(cl, vol).on_balance_volume()
    ar = AroonIndicator(hi, lo, window=25)
    aru = ar.aroon_up()
    ard = ar.aroon_down()
    sar_d = psar.psar_down()
    sar_u = psar.psar_up()
    # x_grad = list(range(data.shape[0]-validation_length))
    # inter_slope = PolyInter(cl, progress_bar=True).values
    # awes = AwesomeOscillatorIndicator(hi, lo, window1=5, window2=34).awesome_oscillator()
    # sar_di = psar.psar_down_indicator()
    # sar_ui = psar.psar_up_indicator()
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
                   # boll_h,
                   grad_bolh,
                   # boll_l,
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
    hi = data[2]
    lo = data[3]
    vol = data[5]
    # op = data[0]

    data = np.array(cl)
    macd = Macd(data, 6, 12, 3).values
    stoch_rsi = StochRsi(data, period=14).hist_values
    grad_rsi = np.gradient(stoch_rsi, list(range(data.shape[0])))
    dpo = Dpo(data, period=4).values
    cop = Coppock(data, wma_pd=10, roc_long=6, roc_short=3).values
    grad_cop = np.gradient(cop, list(range(data.shape[0])))
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
    # sar_di = psar.psar_down_indicator()
    # sar_ui = psar.psar_up_indicator()
    # inter_slope = PolyInter(data, progress_bar=True).values
    for ind in gogo:
        if nan_indices[ind]:
            sar_d[ind] = sar_u[ind]
        sar_d[ind] = (sar_d[ind] - cl[ind])/cl[ind]
    xv = np.array([macd,
                   stoch_rsi,
                   grad_rsi,
                   sar_d,
                   # sar_u,
                   aru - ard,
                   ard,
                   obv,
                   # sar_u,
                   # op,
                   #   high,
                   #   low,
                   # cl,
                   # vol,
                   # boll_h,
                   grad_bolh,
                   # boll_l,
                   grad_boll,
                   grad_cop,
                   dpo,
                   cop
                   ])

    xv = np.transpose(xv)
    return xv[validation_lag:]


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
