import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from scipy.signal import savgol_filter
import sys
import os

import statistics
import pandas as pd
import datetime
sys.path.append(os.path.join('C:/', 'Users', 'Fedot','Downloads','LSTM-Crypto-Price-Prediction','historical_data'))

from get_data import get_data_files
class Genlabels(object):
    def __init__(self, data, window, polyorder=3, graph=False):
        # check for valid parameters
        try:
            if window%2 == 0: raise ValueError('Window length must be an odd positive value')
            if polyorder >= window: raise ValueError('Polyorder must be smaller than windows length')
        except ValueError as error:
            sys.exit('Error: {0}'.format(error))   

        # load historic data from file
        self.hist = data['close']# это лист из средней цены между опеном и клоузом
        self.candles = data# здесь долженбыть датафрейм а не лист
        self.window = window
        self.polyorder = polyorder

        # filter data and generate labels
        self.savgol = self.apply_filter(deriv=0,hist = self.hist)
        self.savgol_deriv = self.apply_filter(deriv=1,hist =self.hist)

        self.labels = self.cont_to_disc()

        #if graph: self.graph()

    def apply_filter(self, deriv,hist):
        # apply a Savitzky-Golay filter to historical prices
        return savgol_filter( window_length=self.window, polyorder=self.polyorder, deriv=deriv,x=hist)

    def cont_to_disc(self):
        # encode label as binary (up/down)
        label = []
        ##for value in self.savgol_deriv:
           # if value >= 0: label.append(1)
            #else: label.append(0)
        ct = self.candles.T
        for i in range(ct.shape[1]):
            if ct[i]['open'] < ct[i]['close']:
                label.append(1)
            else:
                label.append(0)


        return np.array(label)









if __name__ == '__main__':
    start = '20 Mar 2023'
    end = '25 Mar 2024'
    candles = get_data_files(start, end, 60)
    #hist = '../historical_data/hist_mean.npy'
    #data = np.load(hist)
    window_size=25
    maxres=0
    for i in range(25,26,2):
     res=Genlabels(window=i, polyorder=4, data=candles, graph=True)
     bench = res.strategy_bench()[0]
     if bench>maxres:
         maxres=bench
         print(i)


