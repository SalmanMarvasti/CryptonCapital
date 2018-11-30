import pandas as pd
import numpy as np
import urllib, json
from collections import namedtuple
import datetime as dt
import time
from twisted.internet import task
from twisted.internet import reactor

tp = namedtuple('tradingpair', ('name', 'datetime', 'market'))


def convert_to_ndarray(k):
    c = 0
    d = 0
    mya = np.empty((len(k), 2), dtype=np.float64)
    for v in k:
        d = 0
        for j in v:
            if (d < 2):
                mya[c, d] = float(k[c][d])

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
        self.tradewindow = 300
        self.blo_probs = []
        self.alo_probs = []



    def getlatestob(self,sfile, obfile):
        with urllib.request.urlopen(self.updateurl+self.tradingpair) as url:
            current_time = time.time()
            latest_ob= json.loads(url.read().decode())
            print(latest_ob)
            k = latest_ob['bids']
            # [float(y) for y in x]
            self.bids = convert_to_ndarray(k)
            savebids = np.column_stack(self.bids,np.zeros(len(self.bids)) )

            self.asks = convert_to_ndarray(latest_ob['asks'])
            saveasks = np.column_stack(self.asks, np.ones(len(self.asks)))




        with urllib.request.urlopen(self.tradeeurl + self.tradingpair) as url:
            latest_mo = json.loads(url.read().decode())
            print(latest_mo)

            temporders = np.array( [[  float(x['price']), float(x['qty']), int(x['time']) , 1 if x['isBuyerMaker'] else 0]for x in latest_mo] )

            threshold = temporders[:,2]>int((current_time - self.tradewindow) * 1000)
            filtorders = temporders[threshold]
            self.marketorders = filtorders
            print('no of market orders'+str(len(self.marketorders)))
            buy_sum = filtorders[filtorders[:,-1]==0].sum(axis=0)
            sell_sum = filtorders[filtorders[:,-1]==1].sum(axis=0)

        sum_bids = self.bids.sum(axis=0)
        sum_asks = self.asks.sum(axis=0)

        prob_blo_live = (sum_bids[1]-sell_sum[1])/(sum_bids[1])
        prob_alo_live = (sum_asks[1] - buy_sum[1]) / (sum_asks[1])
        np.savetxt(sfile, filtorders, fmt="%20.14f",delimiter=',')
        np.savetxt(obfile, savebids, fmt="%20.14f", delimiter=',')
        np.savetxt(obfile, saveasks, fmt="%20.14f", delimiter=',')

        self.blo_probs.append(prob_blo_live)
        self.alo_probs.append(prob_alo_live)
        return


        # self.trades = np.array([[float(y) for y in x] for x in latest_ob.bids])
                #self.asks = np.array([[float(y) for y in x] for x in latest_ob.asks])





            







if __name__ == "__main__":

    tp.name = "EOSUSDT"
    # tp.secret = "Test@123"
    tp.market = "Binance"
    cr = modellingmanager(tp)
    f = open('marketorders2.csv', 'ab')
    fob = open('orderbooks2.csv', 'ab')
    l = task.LoopingCall(cr.getlatestob,f,fob)
    l.start(30.0)  # call every second

    # l.stop() will stop the looping calls
    reactor.run()
    # blo, slo = cr.getlatestob(f)



