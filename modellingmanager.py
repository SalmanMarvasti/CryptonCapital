import pandas as pd
import numpy as np
import urllib, json
from collections import namedtuple
import datetime as dt
import time
from twisted.internet import task
from twisted.internet import reactor
tp = namedtuple('tradingpair', ('name', 'datetime', 'market'))
from twisted.internet import protocol
from twisted.internet import reactor
import os
import time


def convert_df_bins(d2, bins):
    # d2['bins'] =
    return d2.groupby(pd.cut(d2['price'], bins)).sum()



def convert_to_ndarray(k, ctime: int):
    c = 0
    d = 0
    ctime = ctime * 1000 # ms
    mya = np.empty([len(k), 4]) # , dtype = [('price',np.float64), ('qty',np.float64), ('time',np.int64), ('bidask',np.int64)])
    for v in k:
        d = 0
        for j in v:
            if (d < 2):
                mya[c, d] = float(k[c][d])
            else:
                mya[c, 2] = ctime

            d = d + 1
        c = c + 1

    return mya

class modellingmanager:


    def __init__(self, acct):
        global r
        self.tradingpair = acct.name

        self.datetime = acct.datetime
        self.market = acct.market
        self.latestob = {}
        self.updateurl = "https://api.binance.com/api/v1/depth?symbol="
        self.tradeeurl = "https://api.binance.com/api/v1/trades?symbol="
                # .queue = connredis('redis.pinksphere.com')
        self.bins = []
        self.marketorders = []
        self.tradewindow_sec = 300
        self.blo_probs = []
        self.alo_probs = []
        self.tick = 0.01
        self.vwap = -1.0
        self.df = pd.DataFrame()

    def choosetick(self, x: float, y : float):

        baspread = (self.asks[0,0] - self.bids[0,0])*3
        if (x<baspread):
            best = x
        if y<baspread:
            best = y

        print('doane_tick' + str(x) + ' sprd' + str(baspread) + ' fdtick' + str(y))

        return best

    def getlatestob(self,sfile, obfile, pfile):
        with urllib.request.urlopen(self.updateurl+self.tradingpair) as url:

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


        with urllib.request.urlopen(self.tradeeurl + self.tradingpair) as url:
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
        #self.spread = (self.asks[0,0] - self.bids[0,0])
        self.mid = (self.asks[0,0] + self.bids[0,0])*0.5
        orderbook_pricelevels = np.concatenate((self.bids[:,0], self.asks[:,0]))
        self.bins = np.histogram_bin_edges(orderbook_pricelevels, bins='fd')
        dfbids = pd.DataFrame({'price':self.bids[:,0] - self.mid, 'amount':self.bids[:,1],})
        dfasks = pd.DataFrame({'price': self.asks[:, 0]- self.mid, 'amount': self.asks[:, 1], })
        self.tick = self.bins[1]-self.bins[0]
        self.bins = np.histogram_bin_edges(orderbook_pricelevels, bins='doane')
        tick_do = self.bins[1] - self.bins[0]
        self.tick = self.choosetick(tick_do/2, self.tick/3)
        self.bins = np.arange(dfbids['price'].min(), dfasks['price'].max(), self.tick)
        mdf = pd.concat([dfbids,dfasks], ignore_index=True)
        try:
            nfs = convert_df_bins(mdf, self.bins)
            print(nfs)
        except Exception as e:
            print(e)
        self.df = nfs
        prob_blo_live = (sum_bids[1]-sell_sum[1])/(sum_bids[1])
        prob_alo_live = (sum_asks[1] - buy_sum[1]) / (sum_asks[1])
        np.savetxt(sfile, filtorders, fmt="%20.14f",delimiter=',')
        np.savetxt(obfile, self.bids, fmt="%20.14f", delimiter=',')
        np.savetxt(obfile, self.asks, fmt="%20.14f", delimiter=',')
        nn = np.array([[current_time * 100, prob_blo_live, prob_alo_live]])
        np.savetxt(pfile, nn, fmt="%10.4f", delimiter=',')
        self.blo_probs.append(prob_blo_live)
        self.alo_probs.append(prob_alo_live)
        pfile.flush()
        return


        # self.trades = np.array([[float(y) for y in x] for x in latest_ob.bids])
                #self.asks = np.array([[float(y) for y in x] for x in latest_ob.asks])





            







if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("validpair", nargs='?')
    name = 't'
    try:
        args = parser.parse_args()
        print(args.validpair)
        name = args.validpair
    except Exception as e:
        print("type error: " + str(e))
        name = 'EOSUSDT'
    # tp.secret = "Test@123"
    if name is not None:
        tp.name = name
    else:
        tp.name = 'EOSUSDT'
    tp.market = "Binance"
    cr = modellingmanager(tp)
    fp = open(str(tp.name)+'prob2.csv', 'ab')
    f = open('marketorders2.csv', 'ab')
    fob = open('orderbooks2.csv', 'ab')
    l = task.LoopingCall(cr.getlatestob,f,fob,fp)
    l.start(cr.tradewindow_sec)  # call every tradewindow_sec seconds

    # l.stop() will stop the looping calls
    reactor.run()
    # blo, slo = cr.getlatestob(f)
    print('reactor completed')
    fp.close()
    f.close()
    fob.close()


