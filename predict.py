import numpy as np
import pandas as pd
import time
from keras.models import load_model
from build_data import build_val_data, shape_data
# import matplotlib.pyplot as plt


def predict_val(tag, params):
    validation_length, validation_lag, timesteps, regression, calc_xval = params
    candles = pd.DataFrame(np.load('candles.npy', allow_pickle=True))
    lstm = load_model(f'best_opt_model.keras')
    y_strat_p = list()
    y_strat_n = list()
    runs = 0
    select, build, shape = 0, 0, 0
    if calc_xval:

        # y_strat_n.append([])
        # for models in range(1):
        # up, down = False, False
        # lstm = load_model(f'models/lstm_model_{tag}_{models}.h5')

        x_val = np.empty(shape=(1, timesteps, 13))
        val_sample = 10000
        for v in range(validation_length - validation_lag - 2 * timesteps):
            if v > val_sample:
                continue
            t0 = time.time()
            val_input = candles.iloc[-validation_length + v:-validation_length + v + validation_lag + timesteps]
            x_val_single = build_val_data(data=val_input, params=params)
            # graph(X, start_g=31, hold_g=y, len_g=50, col_type=0)
            x_val_single, _ = shape_data(x_val_single, [0], training=False, params=params)
            x_val = np.append(x_val, x_val_single, axis=0)
            t1 = time.time()
            runs += 1
            build += t1-t0
            print(f'adding val data for  candle {v} of {val_sample}')

        x_val = np.delete(x_val, 0, axis=0)
        np.save('x_val', x_val, allow_pickle=True)
    else:
        x_val = np.load("x_val.npy", allow_pickle=True)
    t3 = time.time()
    lstm_pred = lstm.predict(x_val, batch_size=4096)
    t4 = time.time()
    pred_time = t4 - t3
    if regression:
        y_strat_p = lstm_pred[0]
    else:
        y_strat_p = lstm_pred.T[1]


    # close = candles.iloc[-validation_length + v + timesteps][1]
    # open = candles.iloc[-validation_length + v + timesteps][0]
    # re = (close - open) / open
    # if d > 0:

    # u = X_val[0][14][1]
    # y_strat[models].append(u if (np.isnan(d)) else d)
    # lstm_pred[0]
    # np.save(f'X_val', X_val, allow_pickle=True)
    print(f' build {build:2f} predict {pred_time:2f} ')
    y_strat_p = pd.DataFrame(y_strat_p)
    y_strat_n = pd.DataFrame(y_strat_n)
    np.save(f'y_strat_{tag}_p', y_strat_p, allow_pickle=True)
    np.save(f'y_strat_{tag}_n', y_strat_n, allow_pickle=True)



def getpreds(tag):
    y_pred = []
    y_strat = np.load(f'y_strat_pos_{tag}.npy', allow_pickle=True)

    for col in range(y_strat.shape[0] - 1):
        column = y_strat[col]
        # for i in range(column.shape[0]-1):
        # if np.isnan(column[i]):
        # column =np.delete(column, i)
        vec_p = np.nanmean(column)

        y_pred.append(vec_p)  # statistics.mean(vec))
        # tempsum = 0
        # sums = []
        # for pr in range(12, len(y_pred)):
        #    tempsum += y_pred[pr]
        #    sums.append(tempsum)

    return y_pred


# np.save("predictions", y_pred, allow_pickle=True)
#   preds = list()
# y_pred = np.load("predictions.npy", allow_pickle=True)
#   for p in y_pred:
#   preds.append(p[0])
#   y_pred=preds
#   k =y_pred[1]
#   y_pred_keras = keras_model.predict(X_test).ravel()
# y_pred_n = np.array(binarypreds(getpreds('neg')))
# y_pred_p = y_pred_p - y_pred_n


def preds2binary(pred):
    if np.isnan(pred):
        return
    for v in range(len(pred)):
        if pred[v] < 0.4:
            pass  # pred[v] = 1
        else:
            if pred[v] > 0.6:
                pass  # pred[v] = 0
            else:
                pass  # pred[v] = None

    return pred


def addnones(pred, params):
    val_lag, timesteps = params[1:3]
    for _ in range(timesteps + val_lag):
        pred = np.insert(pred, 0, axis=0)
    for _ in range(timesteps):
        pred = np.append(pred, np.nan)
    return pred


def makegrad(pred):
    grad = []
    for pr in range(1, pred.shape[0]):
        fragment = pred[pr - 1:pr + 1]
        frag_grad = np.gradient(fragment, [0, 1])
        grad.append(frag_grad[0])
    return grad