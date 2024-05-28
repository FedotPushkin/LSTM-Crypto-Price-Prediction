import numpy as np
import pandas as pd
from keras.models import load_model
from build_data import build_val_data, shape_data
import matplotlib.pyplot as plt


def predict(tag, params):
    validation_length, validation_lag, timesteps = params[:-1]
    calc_xval = False
    candles = pd.DataFrame(np.load('candles.npy', allow_pickle=True))
    if calc_xval:
        y_strat_p = list()
        # y_strat_p.append([])
        y_strat_n = list()
        # y_strat_n.append([])
        # for models in range(1):
        # up, down = False, False
        # lstm = load_model(f'models/lstm_model_{tag}_{models}.h5')

        x_val = list()
        lstm = load_model(f'my_model.keras')
        for v in range(validation_length - validation_lag - 2 * timesteps):
            val_input = candles.iloc[-validation_length + v:-validation_length + v + validation_lag + timesteps]
            x_val_single = build_val_data(data=val_input)
            # graph(X, start_g=31, hold_g=y, len_g=50, col_type=0)
            x_val_single, _ = shape_data(x_val_single, [0], training=False)
            x_val.append(x_val_single)

            lstm_pred = lstm.predict(x_val_single, batch_size=1)
            d = lstm_pred[0]
            y_strat_p.append(d)
            print(f'adding val data for  candle {v} of {validation_length}')
            # close = candles.iloc[-validation_length + v + timesteps][1]
            # open = candles.iloc[-validation_length + v + timesteps][0]
            # re = (close - open) / open
            # if d > 0:

            # u = X_val[0][14][1]
            # y_strat[models].append(u if (np.isnan(d)) else d)
            # lstm_pred[0]
        # np.save(f'X_val', X_val, allow_pickle=True)
        y_strat_p = pd.DataFrame(y_strat_p)
        y_strat_n = pd.DataFrame(y_strat_n)
        np.save(f'y_strat_{tag}_p', y_strat_p, allow_pickle=True)
        np.save(f'y_strat_{tag}_n', y_strat_n, allow_pickle=True)
    else:
        # X_val = np.load(f'X_val.npy', allow_pickle=True)
        print(f'xval loaded')
        # newxval=list()
    # for i in range(X_val.shape[0]):
    # newxval.append(X_val[i][0])
    # newxval = pd.DataFrame( np.array(newxval))


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