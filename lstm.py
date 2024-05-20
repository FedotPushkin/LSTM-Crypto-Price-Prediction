import numpy as np
import pandas as pd
from collections import Counter
from keras.models import load_model
# from verstack.stratified_continuous_split import scsplit
import matplotlib.pyplot as plt
# cross_val_score
from generate_labels import Genlabels
from build_data import build_tt_data, build_val_data, shape_data
from shuffle import shuffle_and_train
from xgboost import XGBRegressor, XGBClassifier
from get_data import get_data_files
from graph import graph

# sys.path.append(os.path.join('C:/', 'Users', 'Fedot', 'Downloads', 'LSTM-Crypto-Price-Prediction', 'historical_data'))
# sys.path.append(os.path.join('historical_data'))
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# @keras.saving.register_keras_serializable()

if __name__ == '__main__':
    start = '01 Jan 2017'
    end = '08 Apr 2024'
    load_data = False
    reshuffle = False
    train = False
    predict = False
    bench = True
    regress = False
    validation_length = 1000
    validation_lag = 30
    timesteps = 15
    if load_data:
        candles = get_data_files(start, end, 60)
        np.save('candles', candles, allow_pickle=True)
        cl = np.load("hist_cl.npy")
        op = np.load("hist_op.npy")
        lo = np.load("hist_lo.npy")
        hi = np.load("hist_hi.npy")
        res = np.load("hist_mean.npy")
        vol = np.load("vol.npy")
        date = np.load("Date.npy")

        candles = pd.DataFrame({'open': op, 'close': cl, 'high': hi, 'low': lo, 'mean': res, 'vol': vol, 'Date': date})
        X = build_tt_data(candles)
        labels_p, labels_n = Genlabels(candles['close'], window=25, polyorder=3).labels
        cp = Counter(labels_p)
        # labels_n = []
        cn = Counter(labels_n)
        print(f'initial pos {cp.most_common(2)}')
        print(f'initial neg {cn.most_common(2)}')
        y_p = labels_p[31:1 - validation_length]
        y_n = labels_n[31:1 - validation_length]
        # graph(X, start_g=31, hold_g=y, len_g=100, col_type=1)
        y_val_p = labels_p[-validation_length:]
        y_val_n = labels_n[-validation_length:]

        np.save('X', X, allow_pickle=True)
        np.save('y_p', y_p, allow_pickle=True)
        np.save('y_n', y_n, allow_pickle=True)
        np.save('y_val_p', y_val_p, allow_pickle=True)
        np.save('y_val_n', y_val_n, allow_pickle=True)
        # np.save('val_labels', [0], allow_pickle=True)
    X, y_p, y_n, y_val_p, y_val_n, candles = np.load('X.npy', allow_pickle=True), \
        np.load('y_p.npy', allow_pickle=True), \
        np.load('y_n.npy', allow_pickle=True), \
        np.load('y_val_p.npy', allow_pickle=True), \
        np.load('y_val_n.npy', allow_pickle=True), \
        np.load('candles.npy', allow_pickle=True)
    xp = X
    xn = X
    xp, y_p = shape_data(xp, y_p, training=True)
    xn, y_n = shape_data(xn, y_n, training=True)
    if train:
        #   ensure equal number of labels, shuffle, and split
        shuffle_and_train(xp, y_p, 'pos', reshuffle=reshuffle)
        # shuffle_and_train(xn, y_n, 'neg')

    #   print(model.summary())
    candles = pd.DataFrame(candles)
    if predict:

        # y_val_p = to_categorical(y_val_p, 2)
        # y_val_n = to_categorical(y_val_n, 2)

        def predict(tag):

            y_strat_p = list()
            #y_strat_p.append([])
            y_strat_n = list()
            #y_strat_n.append([])
            # for models in range(1):
            adding = True
            lencount = 1
            up, down = False, False
            # lstm = load_model(f'models/lstm_model_{tag}_{models}.h5')
            lstm = load_model(f'my_model.keras')
            for v in range(validation_length - validation_lag - 2*timesteps):
                val_input = candles.iloc[-validation_length+v:-validation_length+v+validation_lag+timesteps]
                X_val = build_val_data(data=val_input)
                # graph(X, start_g=31, hold_g=y, len_g=50, col_type=0)
                X_val, _ = shape_data(X_val, [0], training=False)
                lstm_pred = lstm.predict(X_val, batch_size=1)
                d = lstm_pred[0][0]
                y_strat_p.append(d)
                print(f'epoch {v} of 1000')
                #close = candles.iloc[-validation_length + v + timesteps][1]
                #open = candles.iloc[-validation_length + v + timesteps][0]
                #re = (close - open) / open
                #if d > 0:

                # u = X_val[0][14][1]
                # y_strat[models].append(u if (np.isnan(d)) else d)
                # lstm_pred[0]
            y_strat_p = pd.DataFrame(y_strat_p)
            y_strat_n = pd.DataFrame(y_strat_n)
            np.save(f'y_strat_{tag}_p', y_strat_p, allow_pickle=True)
            np.save(f'y_strat_{tag}_n', y_strat_n, allow_pickle=True)


        predict('pos')
        # predict('neg')

    def getpreds(tag):
        y_pred = []
        y_strat = np.load(f'y_strat_pos_{tag}.npy',  allow_pickle=True)

        for col in range(y_strat.shape[0]-1):
            column = y_strat[col]
            # for i in range(column.shape[0]-1):
            # if np.isnan(column[i]):
            # column =np.delete(column, i)
            vec_p = np.nanmean(column)

            y_pred.append(vec_p)    # statistics.mean(vec))
            tempsum = 0
            sums = []
            for pr in range(12, len(y_pred)):
                tempsum += y_pred[pr]
                sums.append(tempsum)

        return y_pred

    # np.save("predictions", y_pred, allow_pickle=True)
    #   preds = list()
    # y_pred = np.load("predictions.npy", allow_pickle=True)
    #   for p in y_pred:
    #   preds.append(p[0])
    #   y_pred=preds
    #   k =y_pred[1]
    #   y_pred_keras = keras_model.predict(X_test).ravel()
    def binarypreds(pred):
        if np.isnan(pred):
            return
        for v in range(len(pred)):
            if pred[v] < 0.4:
                pass    # pred[v] = 1
            else:
                if pred[v] > 0.6:
                    pass    # pred[v] = 0
                else:
                    pass    # pred[v] = None

        return pred

    def addnones(pred):
        for _ in range(timesteps + validation_lag):
            pred = np.insert(pred, 0, None, axis=0)
        for _ in range(timesteps):
            pred = np.append(pred, None)
        return pred
    y_pred_p = np.array(getpreds('p'))
    y_pred_n = np.array(getpreds('n'))
    plt.figure(1)
    plt.plot(y_pred_p)
    plt.title('profit ')
    plt.ylabel('sum roi')
    plt.xlabel('pred value(0.3-0.9')
    plt.show()
    plt.figure(2)
    plt.plot(y_pred_n)
    plt.title('profit ')
    plt.ylabel('sum roi')
    plt.xlabel('pred value(0.3-0.9')
    plt.show()
    # y_pred_n = np.array(binarypreds(getpreds('neg')))
    # y_pred_p = y_pred_p - y_pred_n

    def makegrad(pred):
        grad = []
        for pr in range(1, pred.shape[0]):
            fragment = pred[pr - 1:pr + 1]
            frag_grad = np.gradient(fragment, [0, 1])
            grad.append(frag_grad[0])
        return grad


    #grad_pred_p = makegrad(y_pred_p)
    # grad_pred_n = makegrad(y_pred_n)
    # y_pred_p = np.delete(y_pred_p, y_pred_p.shape[0] - 1)
    # y_pred_n = np.delete(y_pred_n, y_pred_n.shape[0] - 1)
    # x_pred = list(range(y_pred_p.shape[0]))
    # grad_pred_p_full = np.gradient(y_pred_p, x_pred)
    # grad_pred_n = np.gradient(y_pred_n, x_pred)
    # grad_pred_p = np.insert(grad_pred_p, 0, None, axis=0)
    # grad_pred_n = np.insert(grad_pred_n, 0, None, axis=0)
    hit = 0
    miss = 0

    candles_val = candles[-validation_length:-timesteps]
    savgol = Genlabels(candles_val[1], window=25, polyorder=3).apply_filter(deriv=0, hist=candles_val[1])
    savgol_deriv = Genlabels(candles_val[1], window=25, polyorder=3).apply_filter(deriv=1, hist=candles_val[1])
    labels_p, labels_n = Genlabels(candles[1], window=25, polyorder=3).labels
    lag = labels_p.shape[0]-validation_length+validation_lag+timesteps
    for i in range(validation_length - validation_lag - 2 * timesteps-1):

        #if (y_pred_p[i] > 0.5 and savgol_deriv[i+validation_lag+timesteps]>0.5) \
        #        or (y_pred_p[i] < 0.5 and savgol_deriv[i +validation_lag+timesteps] < 0.5):

        if (y_pred_p[i] < 0.4 and labels_p[i+lag] > 0.5) \
                or (y_pred_p[i] > 0.6 and labels_p[i+lag] < 0.5):
            hit += 1
        elif (y_pred_p[i] >= 0.4 and labels_p[i+lag] > 0.5) \
                or (y_pred_p[i] <= 0.6 and labels_p[i+lag] < 0.5):
          miss += 1
    print(f'savg res {hit / (hit + miss)}')
    target = []
    for i in range(validation_lag+timesteps, validation_length-timesteps):
        # target.append((candles[1][i]-candles[0][i])/candles[0][i])
        # target.append((candles[1][i] > candles[0][i]) )
        target.append(1 if savgol_deriv[i] > 0.0 else 0)
    # x_train_xg = np.array([y_pred_p[:-100], y_pred_n[:-100],
    # np.array(grad_pred_p)[:-100], np.array(grad_pred_n)[:-100]]).T
    # x_test_xg = np.array([y_pred_p[-100:], y_pred_n[-100:],
    # np.array(grad_pred_p)[-100:], np.array(grad_pred_n)[-100:]]).T

    def trainxg(train_data, targets):

        # xtrain, xtest, ytrain, ytest = train_test_split(train_data, targets, test_size=.2)
        # create model instance
        bst = XGBClassifier()
        # fit model
        history = bst.fit(train_data, targets)
        print("Training score: ", bst.score(train_data, targets))
        # make predictions
        return bst
    # y_trainxg = target[1:-100]
    # xg = trainxg(x_train_xg, y_trainxg)
    roi_preds = []  # xg.predict(x_test_xg)
    hit, miss = 0, 0
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
    if bench:
        # start = candles.shape[0] - validation_length

        # random_guess(y_pred)
        start = candles.shape[0] - validation_length+validation_lag+timesteps
        # _, __, holding_m = strategy_bench(preds=y_pred_p, start_pos=start, verb=True)
        # np.save("holding", holding_m, allow_pickle=True)
        # start = len(candles) - validation_length
        # np.save("start", [start], allow_pickle=True)
    else:
        start = len(candles[6]) - 100
        holding_m = np.load("holding.npy")
        start = np.load("start.npy")[0]
        start = len(candles[6]) - 100
    # bench_cand(y_pred_p)
    # bench_cand(y_pred_n)
    start = candles.shape[0] - validation_length
    graph(x_g=None, hold_g=[], start_g=start, len_g=200, col_type=0)
