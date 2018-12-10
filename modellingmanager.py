import pandas as pd
import numpy as np
import urllib, json
import pickle
from collections import namedtuple, deque
import datetime as dt
import time
from twisted.internet import task
from twisted.internet import reactor
tp = namedtuple('tradingpair', ('name', 'datetime', 'market'))
from twisted.internet import protocol
from twisted.internet import reactor
import os
import time
from atomicwrites import atomic_write


def create_model(pairname, exchange):
    tp.name = pairname
    tp.market = exchange

    return modelob(tp).loadfrompickle()


def convert_df_bins(d2, bins):
    # d2['bins'] =
    return d2.groupby(pd.cut(d2['price'], bins)).sum()



def convert_to_ndarray(k, ctime, isAsk=0):
    c = 0
    d = 0
    ctime = ctime * 1000 # ms
    mya = np.empty([len(k), 4]) #, dtype = [np.float64, np.float64, np.int64, np.int32]) # , dtype = [('price',np.float64), ('qty',np.float64), ('time',np.int64), ('bidask',np.int64)])
    for v in k:
        d = 0
        for d in range(0,4):
            if (d < 2):
                mya[c, d] = float(k[c][d])
            else:
                if d == 2:
                    mya[c, 2] = ctime
                else:
                    mya[c, d] = isAsk
        c = c + 1

    return mya


FIXTIC = 0.015625

class modelob:
    def __init__(self, acct):
        global r
        self.tradingpair = acct.name

        self.datetime = acct.datetime
        self.market = acct.market
        self.latestob = {}
        self.tradewindow_sec = 300

        if self.market =='Binance':
            self.updateurl = "https://api.binance.com/api/v1/depth?symbol={0}"
            self.tradeeurl = "https://api.binance.com/api/v1/trades?symbol=/{0}"
        if self.market in ('BitMex', 'Bitmex'):
            self.updateurl = "https://www.bitmex.com/api/bitcoincharts/{0}/orderBook"
            self.tradeeurl = "https://www.bitmex.com/api/bitcoincharts/{0}/trades"
            self.tradewindow_sec = 35 # bitmex trade window must be lower as buyer or seller is not specified

        # .redis = connredis('redis.pinksphere.com')
        self.bins = []
        self.marketorders = []
        self.blo_probs = deque([], maxlen=5)
        self.alo_probs = deque([], maxlen=5)
        self.mid = deque([], maxlen=5)
        self.tick = FIXTIC  # 1/64
        self.vwap = -1.0
        self.df = pd.DataFrame()
        self.filepath = './'
        self.forcast_estimate_time = 15
        self.SAVEDEBUG = True


    def loadfrompickle(self):
        file = open(self.filepath +'df/'+self.tradingpair+'dataframe.pkl', 'rb')
        return pickle.load(file)

    def saveobjecttofile(self):
        with atomic_write(self.filepath +'df/'+self.tradingpair+'dataframe.pkl', mode='wb', overwrite=True) as output:
            output.write(pickle.dumps(self, pickle.HIGHEST_PROTOCOL))

    # def __getstate__(self):
    #     state = {
    #         'tradingpair':self.tradingpair,
    #         'blo_probs':self.blo_probs,
    #         'alo_probs':self.alo_probs,
    #         'tick':self.tick,
    #     }
    #     print('__getstate__ -> {!r}'.format(state))
    #     return state
    #
    # def __setstate__(self, state):
    #     self._setVar(state)

    def _setVar(self, var):
        for key, value in var.items():
            setattr(self, key, value)

class modellingmanager(modelob):
    def __init__(self, acct):
        modelob.__init__(self, acct)

    def choosetick(self, x: float, y : float):

        baspread = (self.asks[0,0] - self.bids[0,0])*3
        best=baspread
        if (x<baspread):
            best = x
        if y<baspread:
            best = y

        print('doane_tick' + str(x) + ' sprd' + str(baspread) + ' fdtick' + str(y))
        return best


    def probordercompletion(self, timeframe, isask): # approximation based on linear
        n = timeframe
        if len(self.alo_probs)==0 or len(self.blo_probs)==0:
            return 0
        if isask:
            return (1-np.power(np.mean(self.alo_probs),n))  # here
        else:
            return (1-np.power(np.mean(self.blo_probs),n))

    # def probordercompletion(self, timeframe, isask): # approximation based on linear
    #     n = timeframe/self.tradewindow_sec
    #     i = int(n)
    #     if n<1:
    #         i = 1
    #     if isask:
    #         return n* (1-np.power(np.mean(self.alo_probs),i))  # here
    #     else:
    #         return n* (1-np.power(np.mean(self.blo_probs), i))


    def getlatestob(self,sfile, obfile, pfile):
        with urllib.request.urlopen(self.updateurl.format(self.tradingpair)) as url:

            current_time = time.time()

            latest_ob= json.loads(url.read().decode())
            # with open('limitorders.json', 'w') as outfile:
            #     json.dump(latest_ob, outfile)

            print(latest_ob)
            # CONVERTS DATA TO NDARRAY AND GIVES COL NAMES
            self.bids = convert_to_ndarray(latest_ob['bids'], current_time)
            self.bids[:,3] = np.zeros(len(self.bids))
            #savebids = self.bids
            #savebids = np.column_stack([self.bids,np.zeros(len(self.bids)) ])

            self.asks = convert_to_ndarray(latest_ob['asks'], current_time)
            self.asks[:,3] = np.ones(len(self.asks))
            # saveasks = self.asks


        with urllib.request.urlopen(self.tradeeurl.format(self.tradingpair)) as url:
            latest_mo = json.loads(url.read().decode())
            print(latest_mo)
            # with open('marketorders.json', 'w') as outfile:
            #     json.dump(latest_mo, outfile)
            temporders = np.array( [[  float(x['price']), float(x['qty']), int(x['time']) , 1 if x['isBuyerMaker'] else 0]for x in latest_mo] )
            threshold = temporders[:,2]>int((current_time - self.tradewindow_sec) * 1000)
            filtorders = temporders[threshold]
            self.marketorders = filtorders
            print('no of market orders'+str(len(self.marketorders)))
            buy_sum = filtorders[filtorders[:,-1]==0].sum(axis=0)
            sell_sum = filtorders[filtorders[:,-1]==1].sum(axis=0)
        sum_bids = self.bids.sum(axis=0)
        sum_asks = self.asks.sum(axis=0)
        self.mid = (self.asks[0,0] + self.bids[0,0])*0.5
        orderbook_pricelevels = np.concatenate((self.bids[:,0], self.asks[:,0]))
        self.bins = np.histogram_bin_edges(orderbook_pricelevels, bins='fd')
        dfbids = pd.DataFrame({'price':self.bids[:,0] - self.mid, 'amount':self.bids[:,1],})
        dfasks = pd.DataFrame({'price': self.asks[:, 0]- self.mid, 'amount': self.asks[:, 1], })
        tick = self.bins[1]-self.bins[0]
        self.bins = np.histogram_bin_edges(orderbook_pricelevels, bins='doane')
        tick_do = self.bins[1] - self.bins[0]
        if self.tick==FIXTIC:
            self.tick = self.choosetick(tick_do/2, tick/3)
        self.bins = np.arange(dfbids['price'].min(), dfasks['price'].max(), self.tick)
        mdf = pd.concat([dfbids, dfasks], ignore_index=True)
        try:
            nfs = convert_df_bins(mdf, self.bins)
            print(nfs)
        except Exception as e:
            print(e)
        self.df = nfs

        sell_order_per_sec = sell_sum[1] /self.tradewindow_sec
        buy_order_per_sec = buy_sum[1]/ self.tradewindow_sec

        prob_blo_live = (sum_bids[1]-sell_order_per_sec)/(sum_bids[1])
        prob_alo_live = (sum_asks[1] - buy_order_per_sec) / (sum_asks[1])
        floatfmt = '%30.9f'
        timefmt = '%30.3f'
        intfmt = '%i'
        mfmt = [floatfmt, floatfmt, timefmt, intfmt]
        if self.SAVEDEBUG:
            np.savetxt(sfile, filtorders, fmt="%30.10f",delimiter=',')
            np.savetxt(obfile, self.bids, fmt=mfmt, delimiter=',')
            np.savetxt(obfile, self.asks, fmt=mfmt, delimiter=',')
            nn = np.array([[current_time * 1000, prob_blo_live, prob_alo_live]])
            np.savetxt(pfile, nn, fmt="%10.4f", delimiter=',')
            pfile.flush()
            sfile.flush()
        self.blo_probs.append(prob_blo_live)
        self.alo_probs.append(prob_alo_live)
        self.saveobjecttofile()
        return



class bitmexmanager(modellingmanager):
    def __init__(self, acct):
        modelob.__init__(self, acct)



    def getlatestob(self,sfile, obfile, pfile):
        with urllib.request.urlopen(self.updateurl.format(self.tradingpair)) as url:

            current_time = time.time()

            latest_ob= json.loads(url.read().decode())

            print(latest_ob)
            # CONVERTS DATA TO NDARRAY AND GIVES COL NAMES
            self.bids = convert_to_ndarray(latest_ob['bids'], current_time)
            # self.bids[:,3] = np.zeros(len(self.bids))
            #savebids = self.bids
            #savebids = np.column_stack([self.bids,np.zeros(len(self.bids)) ])

            self.asks = convert_to_ndarray(latest_ob['asks'], current_time, 1)
            # saveasks = self.asks

        with urllib.request.urlopen(self.tradeeurl.format(self.tradingpair)) as url:
            latest_mo = json.loads(url.read().decode())
            print(latest_mo)
            # with open('marketorders.json', 'w') as outfile:
            #     json.dump(latest_mo, outfile)
            mid = (self.asks[0,0] + self.bids[0,0])*0.5
            temporders = np.array( [[  float(x['price']), float(x['amount']), int(x['date']) , 1 if x['price']<mid else 0]for x in latest_mo] )
            threshold = temporders[:,2]>int((current_time - self.tradewindow_sec) * 1000)
            filtorders = temporders[threshold]
            self.marketorders = filtorders
            print('no of market orders'+str(len(self.marketorders)))
            buy_sum = filtorders[filtorders[:,-1]==0].sum(axis=0)
            sell_sum = filtorders[filtorders[:,-1]==1].sum(axis=0)
        sum_bids = self.bids[0:6, :].sum(axis=0)
        sum_asks = self.asks[0:6, :].sum(axis=0)
        self.mid = mid

        if self.tick==FIXTIC:
            self.tick = 0.5 # self.choosetick(tick_do/2, 0.5)
        dfbids = pd.DataFrame({'price': self.bids[:,0] - self.mid, 'amount':self.bids[:,1],})
        dfasks = pd.DataFrame({'price': self.asks[:, 0]- self.mid, 'amount': self.asks[:, 1], })
        self.bins = np.arange(dfbids['price'].min(), dfasks['price'].max(), self.tick)
        mdf = pd.concat([dfbids, dfasks], ignore_index=True)
        try:
            nfs = convert_df_bins(mdf, self.bins)
            print(nfs)
        except Exception as e:
            print(e)
        self.df = nfs

        sell_order_per_sec = sell_sum[1] / self.tradewindow_sec
        buy_order_per_sec = buy_sum[1] / self.tradewindow_sec

        prob_blo_live = (sum_bids[1] - sell_order_per_sec) / (sum_bids[1])
        prob_alo_live = (sum_asks[1] - buy_order_per_sec) / (sum_asks[1])
        if prob_alo_live<0:
            prob_alo_live = (1 + np.mean(alo_probs)) * 0.5
        if prob_blo_live<0:
            prob_blo_live = (1 + np.mean(alo_probs)) * 0.5

        floatfmt = '%30.9f'
        timefmt = '%30.3f'
        intfmt = '%i'
        mfmt = [floatfmt,floatfmt, timefmt, intfmt]
        self.blo_probs.append(prob_blo_live)
        self.alo_probs.append(prob_alo_live)
        if self.SAVEDEBUG:
            np.savetxt(sfile, filtorders, fmt="%30.10f",delimiter=',')
            np.savetxt(obfile, self.bids, fmt=mfmt, delimiter=',')
            np.savetxt(obfile, self.asks, fmt=mfmt, delimiter=',')
            nn = np.array([[current_time * 1000, prob_blo_live, prob_alo_live, self.probordercompletion(self.forcast_estimate_time ,0),  self.probordercompletion(self.forcast_estimate_time ,1) ]])
            np.savetxt(pfile, nn, fmt="%10.4f", delimiter=',')
            pfile.flush()
            sfile.flush()
        self.saveobjecttofile()
        return


        # self.trades = np.array([[float(y) for y in x] for x in latest_ob.bids])
                #self.asks = np.array([[float(y) for y in x] for x in latest_ob.asks])



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("validpair", nargs='?')
    parser.add_argument("exchange", nargs='?')
    name = 't'
    exchange = 'Bitmex'
    try:
        args = parser.parse_args()
        print(args.validpair)
        name = args.validpair
        exchange = args.exchange
    except Exception as e:
        print("type error: " + str(e))
        name = 'EOSUSDT'
    # tp.secret = "Test@123"
    if exchange is None:
        print('setting default exchange Bitmex')
        exchange = 'Bitmex'
    if name is not None:
        tp.name = name
    else:
        tp.name = 'XBTUSD' # BTSUSDC for other exchanges

    tp.market = exchange
    if exchange == 'Bitmex':
        cr = bitmexmanager(tp)
    else:
        cr = modellingmanager(tp)
    prefix = './csv/'
    fp = open(prefix+str(tp.name)+'prob2.csv', 'ab')
    f = open(prefix+str(tp.name)+'marketorders2.csv', 'ab')
    fob = open(prefix+str(tp.name)+'orderbooks2.csv', 'ab')
    l = task.LoopingCall(cr.getlatestob,f,fob,fp)
    l.start(cr.tradewindow_sec-10)  # call every tradewindow_sec seconds

    # l.stop() will stop the looping calls
    reactor.run()
    # blo, slo = cr.getlatestob(f)
    print('reactor completed')
    fp.close()
    f.close()
    fob.close()


