import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
# import plotly.graph_objs as go
import copy
import math

def strategy_bench(preds, start_pos, verb=False, deltab=0, deltas=0):

    start_balance = 1000
    start_short_balance = 1000
    balance = start_balance
    profit_long = []
    leftover = 0
    short_leftover = 0
    profit_short = []
    amount = 0
    amount_short = 0
    trades = 0
    trades_short = 0
    fee = 0.02/100
    # 'long':1,'short':-1,'wait:0
    position = 0
    holding = list()
    candles = pd.DataFrame(np.load('candles.npy', allow_pickle=True))
    for i in range(start_pos, start_pos+preds.size-1):
        #   self.savgol = self.apply_filter(deriv=0, hist=candles.iloc[:i + 1]['mean'])
        #   self.savgol_deriv = self.apply_filter(deriv=1, hist=candles.iloc[:i + 1]['mean'])
        #   ####long#####
        if preds[i - start_pos] is None:
            raise Exception('some prediction is none')
        if balance > 0 and amount == 0 and position == 0 and preds[i-start_pos] > 0.5+deltab:
            #   #open long#
            #   amount available after opening long position
            amount = start_balance/(candles.iloc[i][0]*(1+fee))
            balance = 0
            position = 1
            holding.append(1)
            if verb:
                print('long at '+str(candles.iloc[i][6])+' for ' + str(candles.iloc[i][0]))

        else:
            # #close long#
            if amount > 0 and position == 1 and balance == 0 and (preds[i-start_pos] < 0.5-deltas):

                profit_long.append(((1-fee)*amount*candles.iloc[i][0]-start_balance)/start_balance)
                leftover += (1-fee)*amount*candles.iloc[i][0]-start_balance
                trades += 1
                balance = start_balance
                amount = 0
                position = 0
                holding.append(0)
                if verb:
                    print('short at ' + str(candles.iloc[i][6]) + ' for ' + str(candles.iloc[i][0]))
            #   ####short#####
            else:
                #   #open short#

                if position == 0 and amount == 0 and amount_short == 0 and preds[i-start_pos] < 0.5-deltas:
                    position = -1
                    # how much was sold
                    amount_short = start_short_balance/(candles.iloc[i][0]*(1 - fee))
                    holding.append(-1)
                    if verb:
                        print('short at ' + str(candles.iloc[i][6]) + ' for ' + str(candles.iloc[i][0]))
                else:
                    #   #close short#
                    if position == - 1 and preds[i-start_pos] > 0.5-deltab and amount == 0 and amount_short > 0:
                        position = 0
                        curr_profit = start_short_balance-(1 + fee) * amount_short * candles.iloc[i][0]
                        short_leftover += curr_profit
                        profit_short.append(curr_profit/start_short_balance)
                        amount_short = 0
                        holding.append(0)
                        trades_short += 1
                        if verb:
                            print('long at ' + str(candles.iloc[i][6]) + ' for ' + str(candles.iloc[i][0]))
                    else:
                        holding.append(position)
        #   self.savgol = self.apply_filter(deriv=0, hist=candles.iloc[:i + 3]['mean'])
        #   self.savgol_deriv = self.apply_filter(deriv=1, hist=candles.iloc[:i + 3]['mean'])
    if 1:
        print('balance ', balance, ' amount ', amount, ' trades ', trades,
              ' amount short ', amount_short, ' trades_short ', trades_short, ' deltab', deltab, ' deltas', deltas)
    result = ((balance+leftover+amount*(1-fee)*candles[1][candles.shape[0]-1])/start_balance)-1
    if amount_short > 0:
        short_leftover += start_short_balance - amount_short * (1+fee)*candles[1][candles.shape[0]-1]
    result_short = short_leftover/start_short_balance
    if verb:
        if amount > 0:
            print('liqudating long at  ' + str(candles[2][candles.shape[0]-1]))
        if amount_short > 0:
            print('liqudating short at  ' + str(candles[2][candles.shape[0]-1]))
    # y_real = y_val_p[validation_lag+timesteps:-timesteps]
    # guessed_right = 0
    #    diffs = []
    for g in range(preds.size):
        pass
        # if preds[g] == y_real[g]:
        #   guessed_right += 1
        # diffs.append(abs(preds[g]-y_real[g]))
    #   print(f'guessed right {(100*guessed_right/len(preds)):.2f}%')
    #   print('threshhold',thresl,'trades long',trades,',trades_short', trades_short)
    #   print('result_long %.2f avg  long %.2f result_short %.2f avg  short %.2f'%
    #   (result,statistics.mean(profit_long),result_short,statistics.mean(profit_short)))
    trace1 = px.histogram(profit_long, nbins=400, title='longs')

    trace2 = px.histogram(profit_short, nbins=400, title='shorts')
    print(f' avg profit long {np.mean(profit_long)}')
    print(f' avg profit short {np.mean(profit_short)}')
    # trace3 = px.histogram(diffs, nbins=50, title='diffs')
    prof = 0
    for p in profit_long:
        if p > 0:
            prof += 1
    if len(profit_long) > 0:
        print('winrate long ' + str(prof/len(profit_long)))
    prof = 0
    for p in profit_short:
        if p > 0:
            prof += 1
    if len(profit_short) > 0:
        pass
        #  print('winrate short ' + str(prof / len(profit_short)))
    # trace1.show()
    # trace2.show()
    # trace3.show()

    def sign(a):
        if a > 0:
            return '+'
        else:
            return''
    result_str = f'long {sign(result)} {result*100:.2f}%, short {sign(result_short)}{result_short*100:.2f}%'
    print(result_str)
    return result, result_short, holding


def random_guess(length, start, preds):
    # launches strategybench many times on random long/short orders and shows average result
    res = []
    res_s = []

    for _ in range(10):
        avg_hold_dur = 6
        y_rand = list()
        # for o in range(0, length, avg_hold_dur):
        #    n = np.random.random_integers(low=0, high=1)
        #    for k in range(avg_hold_dur):
        #        y_rand.append(n)

        a, b, _ = strategy_bench(preds=np.array(spoil_labels(preds, 45)), start_pos=start, verb=False)
        res.append(a)
        res_s.append(b)
    print('mean long', np.mean(res), 'mean short', np.mean(res_s))


def spoil_labels(labels, rate):
    spoiled = copy.deepcopy(labels)
    for i in range(labels.size):
        r = np.random.random_integers(low=1, high=100)
        if r < rate:
            spoiled[i] = (labels[i]+1) % 2
    return spoiled


def improve_labels(labels, true_labels, rate):
    improved_l = copy.deepcopy(labels)
    improve_count = math.ceil(rate*labels.size/100)
    i = 0
    points = set()
    while i < improve_count:
        point = np.random.random_integers(low=0, high=labels.size-1)
        if abs(labels[point]-true_labels[point]) > 0.5 and point not in points:
            improved_l[point] = true_labels[point]
            points.add(point)
            i += 1
    return improved_l

def bench_cand(pred, timesteps):
    candles = np.load('candles.npy', allow_pickle=True)
    start_pos = len(candles)-timesteps-100

    profit = list()
    # for j in range(0, 101):
    temp_profit = 0
    for s in range(start_pos, start_pos + len(pred) - 1):
        if pred[s-start_pos] is None:
            pass
        elif pred[s-start_pos]:
            temp_profit += ((1-0.001)*candles[1][s]-(1+0.001)*candles[0][s])/candles[0][s]
    profit.append(temp_profit)
    plt.figure(1)
    plt.plot(profit)
    plt.title('profit ')
    plt.ylabel('sum roi')
    plt.xlabel('pred value(0.3-0.9')
    plt.show()
