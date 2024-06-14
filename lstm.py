import numpy as np
import pandas as pd
from collections import Counter
# from verstack.stratified_continuous_split import scsplit
# cross_val_score
from generate_labels import Genlabels
import matplotlib.pyplot as plt
from build_data import build_tt_data, shape_data
from shuffle import shuffle_and_train, plotauc
from xgboost import XGBClassifier  # XGBRegressor
import math
from ta.trend import sma_indicator
from graph import graph
from strategy_benchmark import strategy_bench, random_guess, improve_labels
from predict import getpreds, predict_val
from get_data import get_data_files
# sys.path.append(os.path.join('C:/', 'Users', 'Fedot', 'Downloads', 'LSTM-Crypto-Price-Prediction', 'historical_data'))
# sys.path.append(os.path.join('historical_data'))
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# @keras.saving.register_keras_serializable()

if __name__ == '__main__':
    start = '01 Jan 2017'
    end = '07 June 2024'
    load_data = True
    reshuffle = True
    train = True
    train_model = False
    use_checkpoints = False
    predict = True
    calc_xval = True
    bench = True
    regression = False
    validation_length = 20000
    validation_lag = 28
    timesteps = 15
    labels_wnd = 25
    features = 21
    long, med, short = 26, 12, 9
    hl, ll, sl, ss = math.ceil(long / 2), math.ceil(long * 1.27), math.ceil(long * 0.77), math.ceil(short / 2)
    params = [validation_length, validation_lag, timesteps, features, regression, calc_xval, reshuffle, use_checkpoints, train_model, long, med, short, hl, ll, sl, ss]
    if load_data:
        #candles = get_data_files(start, end, 15)
        #np.save('candles', candles, allow_pickle=True)
        cl = pd.Series(np.load("hist_cl.npy"))
        op = np.load("hist_op.npy")
        lo = np.load("hist_lo.npy")
        hi = np.load("hist_hi.npy")
        res = np.load("hist_mean.npy")
        vol = pd.Series(np.load("vol.npy"))
        date = np.load("Date.npy")
        def plot_price_to_volume():
            sma_vol = pd.Series(sma_indicator(vol, window=20, fillna=False))
            sma_cl = pd.Series(sma_indicator(cl, window=20, fillna=False))
            plt.subplot(2, 1, 1)
            plt.plot(sma_vol)

            plt.ylabel('vol')
            plt.xlabel('time')

            plt.subplot(2, 1, 2)
            plt.plot(sma_cl)
            plt.ylabel('price')
            plt.xlabel('time')
            plt.show()
        candles = pd.DataFrame({'open': op, 'close': cl, 'high': hi, 'low': lo, 'mean': res, 'vol': vol, 'Date': date})

        X = build_tt_data(candles, params)
        labels_p, labels_n = Genlabels(candles['close'], window=labels_wnd, polyorder=3).labels
        if regression:
            cp = Counter(num > 0 for num in labels_p)

            print(f'initial positive examples % {100*cp[True]/(cp[True]+cp[False]):.2f}')
        else:
            cp = Counter(labels_p)
            # cn = Counter(labels_n)
            # print(f'initial pos {cp.most_common(2)}')
            # print(f'initial neg {cn.most_common(2)}')

        y_p = labels_p[validation_lag+timesteps:- validation_length]
        y_n = labels_n[validation_lag+timesteps:- validation_length]

        xp = shape_data(X, training=True, timesteps=timesteps)
        np.save('X_t15_l25_f21_s', xp, allow_pickle=True)
        np.save('y_p', y_p, allow_pickle=True)
        np.save('y_n', y_n, allow_pickle=True)
        # np.save('y_val_p', y_val_p, allow_pickle=True)
        # np.save('y_val_n', y_val_n, allow_pickle=True)
        # np.save('val_labels', [0], allow_pickle=True)
    xp, y_p, y_n, candles = np.load('X_t15_l25_f21_s.npy', allow_pickle=True), \
        np.load('y_p.npy', allow_pickle=True), \
        np.load('y_n.npy', allow_pickle=True), \
        np.load('candles.npy', allow_pickle=True)

    # cl = np.load("hist_cl.npy")
    # labels_p, labels_n = Genlabels(cl, window=25, polyorder=3).labels
    # y_p = labels_p[validation_lag + timesteps:- validation_length]
    # y_n = labels_n[validation_lag + timesteps:- validation_length]
    # xn, y_n = shape_data(xn, y_n, training=True)
    if train:

        #   ensure equal number of labels, shuffle, and split
        shuffle_and_train(xp, y_p, 'pos', params)
        # shuffle_and_train(xn, y_n, 'neg')

    #   print(model.summary())
    candles = pd.DataFrame(candles)
    if predict:
        # y_val_p = to_categorical(y_val_p, 2)
        # y_val_n = to_categorical(y_val_n, 2)

        predict_val(1, params)
    y_pred_p = np.array(getpreds('p'))
    # y_pred_n = np.array(getpreds('n'))

    # grad_pred_p = makegrad(y_pred_p)
    # grad_pred_n = makegrad(y_pred_n)
    # y_pred_p = np.delete(y_pred_p, y_pred_p.shape[0] - 1)
    # y_pred_n = np.delete(y_pred_n, y_pred_n.shape[0] - 1)
    # x_pred = list(range(y_pred_p.shape[0]))
    # grad_pred_p_full = np.gradient(y_pred_p, x_pred)
    # grad_pred_n = np.gradient(y_pred_n, x_pred)
    # grad_pred_p = np.insert(grad_pred_p, 0, None, axis=0)
    # grad_pred_n = np.insert(grad_pred_n, 0, None, axis=0)
    hitp = 0
    missp = 0
    hitn = 0
    missn = 0

    candles_val = candles[-validation_length:-timesteps]
    savgol = Genlabels(candles_val[1], window=labels_wnd, polyorder=3).apply_filter(deriv=0, hist=candles_val[1])
    savgol_deriv = Genlabels(candles_val[1], window=labels_wnd, polyorder=3).apply_filter(deriv=1, hist=candles_val[1])
    labels_p, labels_n = Genlabels(candles[1], window=labels_wnd, polyorder=3).labels
    lag = labels_p.shape[0]-validation_length+validation_lag+timesteps
    balp = 0
    baln = 0
    preds_len = validation_length-validation_lag-timesteps-1

    for i in range(preds_len):  # validation_length - validation_lag - 2 * timesteps-1):

        # if (y_pred_p[i] > 0.5 and savgol_deriv[i+validation_lag+timesteps]>0.5) \
        #        or (y_pred_p[i] < 0.5 and savgol_deriv[i +validation_lag+timesteps] < 0.5):
        delta = 0.02
        if y_pred_p[i] < 0.5-delta:
            baln += 1
            if labels_p[i+lag] == 0:
                hitn += 1
            else:
                missp += 1
        elif y_pred_p[i] > 0.5+delta:
            balp += 1
            if labels_p[i+lag] == 1:
                hitp += 1
            else:
                missn += 1

    plotauc(y_pred_p[:preds_len], labels_p[lag:lag+preds_len], 'check data')
    precision_p = hitp / (hitp + missp)
    recall_p = hitp / (hitp + missn)
    precision_n = hitn / (hitn + missn)
    recall_n = hitn / (hitn + missp)
    f1_score_p = 2 * (precision_p * recall_p) / (precision_p + recall_p)
    f1_score_n = 2 * (precision_n * recall_n) / (precision_n + recall_n)
    print(f'Recall p  {recall_p:.2f}')  #
    print(f'Recall n {recall_n:.2f}')
    print(f'Precision p  {precision_p:.2f}')  #
    print(f'Precision n {precision_n:.2f}')
    print(f'f1 p {f1_score_p:.2f}')
    print(f'f1 n {f1_score_n:.2f}')
    print(f'preds bal {baln / (balp + baln):.2f} zeros')
    cp = Counter(labels_p[lag:lag+preds_len])
    # cn = Counter(labels_n)
    zeros = cp.most_common(2)[0][1]
    ones = cp.most_common(2)[1][1]
    print(f'distrib {cp.most_common(2)} zeros {zeros/(zeros+ones):.2f} ones {ones/(zeros+ones):.2f}')
    target = []
    # for i in range(validation_lag+timesteps, validation_length-timesteps):
    # target.append((candles[1][i]-candles[0][i])/candles[0][i])
    # target.append((candles[1][i] > candles[0][i]) )
    # target.append(1 if savgol_deriv[i] > 0.0 else 0)
    # x_train_xg = np.array([y_pred_p[:-100], y_pred_n[:-100],
    # np.array(grad_pred_p)[:-100], np.array(grad_pred_n)[:-100]]).T
    # x_test_xg = np.array([y_pred_p[-100:], y_pred_n[-100:],
    # np.array(grad_pred_p)[-100:], np.array(grad_pred_n)[-100:]]).T

    def trainxg(train_data, targets):

        # xtrain, xtest, ytrain, ytest = train_test_split(train_data, targets, test_size=.2)
        # create model instance
        bst = XGBClassifier()
        # fit model
        # history = bst.fit(train_data, targets)
        # print("Training score: ", bst.score(train_data, targets))
        # make predictions
        return bst
    # y_trainxg = target[1:-100]
    # xg = trainxg(x_train_xg, y_trainxg)
    roi_preds = []  # xg.predict(x_test_xg)
    # hit, miss = 0, 0
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
    start = candles.shape[0] - validation_length
    if bench:
        # start = candles.shape[0] - validation_length

        # random_guess(y_pred)
        # random_guess(labels_p[lag:].size, start+validation_lag+timesteps, labels_p[lag:])
        best = -10000000
        hold_best, holding_m = [], []
        best_params = [0, 0]
        #y_improved = improve_labels(y_pred_p, labels_p[lag:], 5)
        for i in range(4, 5):
            for j in range(4, 5):
                #y_improved = improve_labels(y_pred_p, labels_p[lag:], 3)
                result, __, holding_m = strategy_bench(preds=y_pred_p,
                                                       # preds=labels_p[lag:],
                                                       start_pos=start+validation_lag+timesteps,
                                                       verb=False,
                                                       deltab=0.2,
                                                       deltas=0.2)
                if result > best:
                    best = result
                    hold_best = holding_m
                    best_params[0] = i
                    best_params[1] = j
        print(f'best score at {best_params}, result {best}')
        np.save("holding", hold_best, allow_pickle=True)
        # start = len(candles) - validation_length
        # np.save("start", [start], allow_pickle=True)
    else:

        hold_best = np.load("holding.npy")
        # start = np.load("start.npy")[0]

    # bench_cand(y_pred_p)
    # bench_cand(y_pred_n)
    lines = [savgol, savgol_deriv, y_pred_p, labels_p[lag:], holding_m]
    #graph(lines=lines, start_g=start, lag=0, col_type=0, params=params)
    #graph(lines=lines, start_g=start, lag=200, col_type=0, params=params)
    # graph(lines=lines, start_g=start, lag=400, col_type=0, params=params)
