import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from scipy.signal import savgol_filter
import sys
import os
import plotly.express as px
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
        self.hist = data.iloc[:window]['mean']
        self.candles = data.iloc[:window]
        self.window = window
        self.polyorder = polyorder

        # filter data and generate labels
        self.savgol = self.apply_filter(deriv=0,hist = self.hist)
        self.savgol_deriv = self.apply_filter(deriv=1,hist =self.hist)

        self.labels = self.cont_to_disc()

        if graph: self.graph()

    def apply_filter(self, deriv,hist):
        # apply a Savitzky-Golay filter to historical prices
        return savgol_filter( window_length=self.window, polyorder=self.polyorder, deriv=deriv,x=hist)

    def cont_to_disc(self):
        # encode label as binary (up/down)
        label = []
        for value in self.savgol_deriv:
            if value >= 0: label.append(1)
            else: label.append(0)
        
        return np.array(label)





    def graph(self):
        # graph the labels
        #candles=pd.read_pickle('../historical_data/candles.csv')
        maxres=0
        pars=[0.4,0.0]
        for i in range(3):
            for j in range(3):
                pass

        holding = self.strategy_bench(thresl=0.4, thress=0.2 )[1]

        trace0 =go.Candlestick(x=candles['Date'],
                                                open=candles['open'],
                                                high=candles['high'],
                                                low=candles['low'],
                                                close=candles['close'])
        trace1 = go.Scatter(x=candles['Date'],y=self.savgol, name='Filter')
        trace2 = go.Scatter(x=candles['Date'],y=self.savgol_deriv, name='Derivative', yaxis='y2')
        trace3 = go.Scatter(x=candles['Date'], y=holding,name='holding', yaxis='y3')
        data = [trace0, trace1, trace2,trace3]

        layout = go.Layout(
            title='Labels',
            yaxis=dict(
                title='USDT value'
            ),
            yaxis2=dict(
                title='Derivative of Filter',
                overlaying='y',
                side='right'
            ),
            yaxis3 = dict(
                title='holding',
                overlaying='y',
                side='right'
            )
        )

        fig2 = go.Figure(data=data, layout=layout)
        #fig2 = go.Figure(data=trace0)#, layout=layout)
        #py.plot(fig1, filename='../docs/label1.html')
        py.plot(fig2, filename='../docs/label2.html')

    def strategy_bench(self,thresl=0.6,thress=0.1,verb=False):

        start_balance=1000
        start_short_balance = 1000
        short_balance=start_short_balance
        balance = start_balance
        profit_long=[]
        leftover=0
        profit_short=[]
        long_value=0
        short_value=0
        amount =0
        amount_short=0
        trades=0
        trades_short=0
        fee=0.001
        position = 0#'long':1,'short':-1,'wait:0
        holding=[]
        for i in range(self.window-1,candles.shape[0]-1):
            self.savgol = self.apply_filter(deriv=0, hist=candles.iloc[:i + 1]['mean'])
            self.savgol_deriv = self.apply_filter(deriv=1, hist=candles.iloc[:i + 1]['mean'])
            #####long#####
            if balance>0 and amount==0 and position ==0 and  self.savgol_deriv[i]>thresl:
                ##open long
                long_value=start_balance
                amount = start_balance*(1-fee)/candles.iloc[i+1]['open']
                balance = 0
                position = 1
                holding.append(1)
                if verb: print('long at'+str(candles.iloc[i+1]['Date'])+'for '+ str(candles.iloc[i+1]['open']))

            else:
                  ##close long
                if self.savgol_deriv[i]<thress and amount>0 and position==1:
                    profit_long.append(((1-fee)*amount*candles.iloc[i+1]['open']-long_value)/long_value)
                    leftover += (1-fee)*amount*candles.iloc[i+1]['open']-start_balance
                    trades+=1
                    balance=start_balance
                    amount = 0
                    position = 0
                    holding.append(0)
                    if verb: print('short at' + str(candles.iloc[i + 1]['Date']) + 'for ' + str(candles.iloc[i + 1]['open']))
                #####short#####
                else:
                    #open short

                    if position==0 and self.savgol_deriv[i]<0-thress:
                        position=-1
                        amount_short=short_balance/candles.iloc[i + 1]['open']
                        short_value = (1 - fee) * amount_short * candles.iloc[i + 1]['open']
                        holding.append(-1)
                    else: #close short
                        if self.savgol_deriv[i] > 0-thress and position==-1:
                            position = 0

                            profit_short.append((short_value-(1 + fee) * amount_short * candles.iloc[i + 1]['open'])/short_value)

                            holding.append(0)
                            trades_short+=1

                        else:
                            holding.append(position)
            #self.savgol = self.apply_filter(deriv=0, hist=candles.iloc[:i + 3]['mean'])
            #self.savgol_deriv = self.apply_filter(deriv=1, hist=candles.iloc[:i + 3]['mean'])


        if verb: print('balance',balance,'amount',amount,'trades',trades,'trades_short ',trades_short)
        result = (leftover+amount*candles.iloc[candles.shape[0]-1]['close'])/start_balance
        result_short = (short_balance + amount_short * candles.iloc[candles.shape[0] - 1]['close']) / start_short_balance
        print('threshhold',thresl,'result_long',result,'trades',trades,'avg % ',statistics.mean(profit_long),' result',result_short,'trades_short',trades_short)

        #trace1 = px.histogram(profit_long,nbins=300)

        #trace2 = px.histogram(profit_short,nbins=400)
        #fig=go.Figure(data=[trace1,trace2])
        #trace2.show()
        return result,holding


if __name__ == '__main__':
    start = '10 Mar 2024'
    end = '25 Mar 2024'
    candles = get_data_files(start, end, 60)
    #hist = '../historical_data/hist_mean.npy'
    #data = np.load(hist)
    window_size=25
    maxres=0
    for i in range(25,26,2):
     res=Genlabels(window=i, polyorder=3, data=candles, graph=True)
     bench = res.strategy_bench()[0]
     if bench>maxres:
         maxres=bench
         print(i)


