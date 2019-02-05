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
import scipy.stats as stats
# from bitmex_websocket import getBitmexWs
from atomicwrites import atomic_write
from bitmexwebsock import getBitmexWs
import logging
from unittest import mock




def create_model(pairname, exchange):
    tp.name = pairname
    tp.market = exchange

    return modelob(tp).loadfrompickle()


def convert_df_bins(d2, bins):
    # d2['bins'] =
    return d2.groupby(pd.cut(d2['price'], bins)).sum()

def diff_df_on_price(tdf1, tdf2):
    # index should be reset if necessary
    res = tdf1.merge(tdf2, on='price', how='outer', indicator=True)
    res = fillna(value={'amount_y': 0, 'amount_x': 0})
    tdf1['change'] =  res['amount_y'] - res['amount_x']
    return tdf1


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
            self.tradeeurl = "https://api.binance.com/api/v1/trades?symbol={0}"
        if self.market in ('BitMex', 'Bitmex'): #'http://localhost:{port}/users'.format(port=38803)
            self.updateurl = "https://www.bitmex.com/api/bitcoincharts/{0}/orderBook"
            self.tradeeurl = "https://www.bitmex.com/api/bitcoincharts/{0}/trades"
            self.tradewindow_sec = 25 # bitmex trade window must be lower as buyer or seller is not specified
        # .redis = connredis('redis.pinksphere.com')
        self.bins = []
        self.marketorders = []
        self.blo_probs = deque([0.1], maxlen=5)
        self.alo_probs = deque([0.1], maxlen=5)
        self.mid = deque([], maxlen=5)
        self.tick = FIXTIC  # 1/64
        self.vwap = -1.0
        self.df = pd.DataFrame()
        self.filepath = './'
        # static const variables
        self.forcast_estimate_time = 90
        self.probfmt = '%2.9f'
        self.SAVEDEBUG = True
        self.buy_sum=deque([0.0,1], maxlen=4)
        self.sell_sum=deque([0.0,1], maxlen=4)
        logging.debug('{0} modelling manager initialised'.format(self.tradingpair))

    def vol_at_lob(self, num_bins_used, is_buy):
        num_bins_used = 4
        askvol = 0
        bidvol = 0

        for i in range(0, num_bins_used):
            askvol += self.asks[i, 1]
            bidvol += self.bids[i, 1]

        if is_buy:
            return bidvol
        else:
            return askvol


    def fit_gamma(self):
        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(data)
        print(fit_alpha, fit_loc, fit_beta)
        # (5.0833692504230008, 100.08697963283467, 21.739518937816108)

        print(alpha, loc, beta)
        # (5, 100.5, 22)
        return fit_alpha, fit_loc,fit_beta
    def loadfrompickle(self):
        file = open(self.filepath +'df/'+self.tradingpair+'dataframe.pkl', 'rb')
        return pickle.load(file)

    def saveobjecttofile(self):
        with atomic_write(self.filepath +'df/'+self.tradingpair+'dataframe.pkl', mode='wb', overwrite=True) as output:
            output.write(pickle.dumps(self, pickle.HIGHEST_PROTOCOL))


    def _setVar(self, var):
        for key, value in var.items():
            setattr(self, key, value)

class modellingmanager(modelob):
    def __init__(self, acct):
        modelob.__init__(self, acct)

    def get_fmt_list(self, timefmt, probfmt, N):
        nnfmt = [timefmt, timefmt]
        for ix in range(2, N):
            nnfmt.append(probfmt)
        return nnfmt

    def choosetick(self, x: float, y : float):

        baspread = (self.asks[0,0] - self.bids[0,0])*3
        best=baspread
        if (x<baspread):
            best = x
        if y<baspread:
            best = y

        print('doane_tick' + str(x) + ' sprd' + str(baspread) + ' fdtick' + str(y))
        return best

    def probordercompletion2(self, timeframe, isask): # approximation based on linear
        n = timeframe
        if (isask and len(self.alo_probs)==0) or (len(self.blo_probs)==0 and not isask):
            return 0
        if isask:
            n_prod = 1
            for n in self.alo_probs:
                n_prod = n*n_prod
            if timeframe>len(self.alo_probs):
                n_prod = np.power(n_prod,timeframe/len(self.alo_probs))
            return 1-n_prod
        else:
            n_prod = 1
            for n in self.blo_probs:
                n_prod = n * n_prod
            if timeframe > len(self.blo_probs):
                n_prod = np.power(n_prod, timeframe / len(self.blo_probs))
            return 1 - n_prod

    def probordercompletion(self, timeframe, isask): # approximation based on linear
        n = timeframe
        if (isask and len(self.alo_probs)==0) or (len(self.blo_probs)==0 and not isask):
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


    def set_lob_data(current_time, latest_ob ):
        self.bids = convert_to_ndarray(latest_ob['bids'], current_time)
        self.bids[:, 3] = np.zeros(len(self.bids))
        # savebids = self.bids
        # savebids = np.column_stack([self.bids,np.zeros(len(self.bids)) ])

        self.asks = convert_to_ndarray(latest_ob['asks'], current_time)
        self.asks[:, 3] = np.ones(len(self.asks))

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
            self.buy_sum.append(buy_sum)
            self.sell_sum.append(sell_sum)
            buy_sum = np.mean(self.buy_sum)
            sell_sum = np.mean(self.sell_sum)
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
            np.savetxt(obfile, self.bids, fmt=mfmt, delimitebgr=',')
            np.savetxt(obfile, self.asks, fmt=mfmt, delimiter=',')
            nn = np.array([[current_time * 1000, mid, quoted['mid'], prob_blo_live, prob_alo_live,
                            self.probordercompletion(self.forcast_estimate_time, 0),
                            self.probordercompletion(self.forcast_estimate_time, 1)]])
            nnfmt = self.get_fmt_list(timefmt, self.probfmt, nn.shape[1])
            np.savetxt(pfile, nn, fmt=nnfmt, delimiter=',')
            pfile.flush()
            sfile.flush()
        self.blo_probs.append(prob_blo_live)
        self.alo_probs.append(prob_alo_live)
        self.saveobjecttofile()
        return



class bitmexmanager(modellingmanager):
    def __init__(self, acct):
        modellingmanager.__init__(self, acct)



    def getlatestob(self,sfile, obfile, pfile):
        mid = 0
        market_order_sell_sum = [1,1] # default one unit bought and sold
        market_order_buy_sum = [1,1]
        try:
            with urllib.request.urlopen(self.updateurl.format(self.tradingpair)) as url:
                alt_orderbook = ws.order_book()
                current_time = time.time()
                latest_ob= json.loads(url.read().decode())

                print(latest_ob)
                # CONVERTS DATA TO NDARRAY AND GIVES COL NAMES
                self.bids = convert_to_ndarray(latest_ob['bids'], current_time)
                # self.bids[:,3] = np.zeros(len(self.bids))
                #savebids = self.bids
                #savebids = np.column_stack([self.bids,np.zeros(len(self.bids)) ])

                self.asks = convert_to_ndarray(latest_ob['asks'], current_time, 1)
                mid = (self.asks[0,0] + self.bids[0,0])*0.5
        
                # saveasks = self.asks

            with urllib.request.urlopen(self.tradeeurl.format(self.tradingpair)) as url:
                latest_mo = json.loads(url.read().decode())
                print(latest_mo)
                # with open('marketorders.json', 'w') as outfile:
                #     json.dump(latest_mo, outfile)
                quoted = ws.get_ticker()
                if np.abs(quoted['mid']-mid)<5:
                    mid = (quoted['mid'] + mid )*0.5
                #if quoted['mid']>mid:
                #    mid = quoted['mid']
                temporders = np.array( [[  float(x['price']), float(x['amount']), int(x['date']) , 1 if x['price']<mid else 0]for x in latest_mo] )
                threshold = temporders[:,2]>int((current_time - self.tradewindow_sec) * 1000)
                filtorders = temporders[threshold]
                self.marketorders = filtorders
                logging.info('no of market orders'+str(len(self.marketorders)))
                market_order_buy_sum = filtorders[(filtorders[:,-1]==0) & (filtorders[:,0]>self.bids[0, 0]), 0:2].sum(axis=0)
                market_order_sell_sum = filtorders[filtorders[:,-1]==1 & (filtorders[:,0]<self.asks[0, 0]), 0:2].sum(axis=0)
        except urllib.error.HTTPError as detail:
            logging.exception(self.tradingpair + ':' + str(detail))
            if detail.errno in (401,500,404):
                print('exception http')
            return
        except ValueError as ver:
            logging.exception("ValueError"+str(ver))
            return
        except Exception as eer:
            logging.exception("unexpected Error")
            logging.warning('Error, may not recover from this')
            return
        except:
            logging.exception("Unhandled Exception")
            raise

        sum_bids = self.bids[0:5, :].sum(axis=0)
        sum_asks = self.asks[0:5, :].sum(axis=0)
        if mid>0:
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

        sell_order_per_sec = market_order_sell_sum[1] / self.tradewindow_sec
        buy_order_per_sec = market_order_buy_sum[1] / self.tradewindow_sec

        prob_blo_live = (sum_bids[1] - sell_order_per_sec) / (sum_bids[1])
        prob_alo_live = (sum_asks[1] - buy_order_per_sec) / (sum_asks[1])
        if prob_alo_live<0:
            prob_alo_live = (1 + np.mean(self.alo_probs)) * 0.5
        if prob_blo_live<0:
            prob_blo_live = (1 + np.mean(self.blo_probs)) * 0.5
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
            nn = np.array([[current_time * 1000, mid,quoted['mid'], prob_blo_live, prob_alo_live, self.probordercompletion2(self.forcast_estimate_time ,0), self.probordercompletion2(self.forcast_estimate_time ,1) ,self.probordercompletion(self.forcast_estimate_time ,0),  self.probordercompletion(self.forcast_estimate_time ,1) ]])
            nnfmt = self.get_fmt_list(timefmt, self.probfmt, nn.shape[1])
            np.savetxt(pfile, nn, nnfmt, delimiter=',')
            pfile.flush()
            sfile.flush()
        self.saveobjecttofile()
        return





if __name__ == "__main__":
    testmode = True
    import argparse

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='./modellingmanager.log',
                        filemode='w')

    parser = argparse.ArgumentParser()
    parser.add_argument("validpair", nargs='?')
    parser.add_argument("exchange", nargs='?')
    parser.add_argument("testmode", nargs='?')
    name = 't'

    exchange = 'Bitmex'
    try:
        args = parser.parse_args()
        #print(args.validpair)
        name = args.validpair
        exchange = args.exchange
        if args.testmode=='yes':
            testmode = True
        if args.testmode == 'no' or args.testmode=='false':
            testmode = False
        if name is not None:
            print('tradingpair:' + str(name)+' exchange:'+str(exchange))
    except Exception as e:
        print("type error: " + str(e))
        name = 'EOSUSDT'
    # tp.secret = "Test@123"
    if not testmode:
        ws = getBitmexWs()
    with mock.patch('bitmexwebsock.BitMEXWebsocket') as MockBitmexgetWs:
        ws = MockBitmexgetWs()


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
    l.start(cr.tradewindow_sec-5)  # call every tradewindow_sec seconds

    # l.stop() will stop the looping calls
    reactor.run()
    # blo, slo = cr.getlatestob(f)
    print('reactor completed')
    fp.close()
    f.close()
    fob.close()


