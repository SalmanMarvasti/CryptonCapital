import pandas as pd
import numpy as np
import urllib, json
import pickle# 5 as pickle
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
    #res = np.fill(value={'amount_y': 0, 'amount_x': 0})
    tdf1['change'] =  res['amount_y'] - res['amount_x']
    return tdf1

def calc_vwap(bids, asks):
    R = 6# bids.shape[0]
    totalb = np.sum(bids[0:R,1])
    totala = np.sum(asks[0:R,1])
    bid_vwap=0
    ask_vwap=0
    for r in range(0, R):
        bid_vwap += bids[r][0]*bids[r][1]
        ask_vwap += asks[r][0]*asks[r][1]
    return (bid_vwap/totalb, ask_vwap/totala, totala, totalb)

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

class prediction_checker:
    def __init__(self):
        self.predlist = []
        self.number_of_predictions = 0
        self.number_correct = 0
        self.time_start = dt.datetime.utcnow().timestamp()
        self.time_end = 0
        self.prev_mid= 0
        self.tick = 0.5
        self.dollar_gain = 0
    def update(self, price, timestamp, tick):
        newlist = []
        self.tick=tick
        for a in self.predlist:
            if timestamp<a[0]:
                if abs(price-a[1])<tick or (a[2]>0 and price>a[1]) or (a[2]<0 and price<a[1]):
                    self.number_correct+=1
                    self.dollar_gain+=abs(a[2])
                else:
                    newlist.append(a)
            else:
                self.dollar_gain -= abs(a[2])
        self.predlist = newlist

    def __len__(self):
        return self.number_of_predictions
    def add_pred(self, validtill_timestamp, price, diff=0):
        if len(self.predlist)>0 and abs(price-self.predlist[-1][1])<(self.tick*0.25):
            print('ignoring duplicate prediction')
            return
        self.predlist.append((validtill_timestamp, price, diff))
        self.number_of_predictions+=1
        if validtill_timestamp>self.time_end:
            self.time_end=validtill_timestamp


class modelob:
    def __init__(self, acct):
        global r
        self.tradingpair = acct.name

        self.datetime = dt.datetime.now()
        self.market = acct.market
        self.latestob = {}
        self.tradewindow_sec = 29
        self.epsi=0.00000000001
        self.stats = prediction_checker()

        if self.market =='Binance':
            self.updateurl = "https://api.binance.com/api/v1/depth?symbol={0}"
            self.tradeeurl = "https://api.binance.com/api/v1/trades?symbol={0}"
        if self.market in ('bitmex', 'BitMex', 'Bitmex'): #'http://localhost:{port}/users'.format(port=53581)
            self.updateurl = "https://www.bitmex.com/api/bitcoincharts/{0}/orderBook"
            self.tradeeurl = "https://www.bitmex.com/api/bitcoincharts/{0}/trades"
            #
            self.tradewindow_sec = 25 # bitmex trade window must be lower as buyer or seller is not specified
        if self.market.lower() =='bitmexws':
            self.tradeeurl = 'http://localhost:{port}/users'.format(port=53581)
            self.updateurl = "https://www.bitmex.com/api/bitcoincharts/{0}/orderBook"
            self.backup_tradeurl = "https://www.bitmex.com/api/bitcoincharts/{0}/trades"
            self.tradewindow_sec = 25  #

        # .redis = connredis('redis.pinksphere.com')
        self.bins = []
        self.marketorders = []
        self.blo_probs = deque([0.99, 0.9], maxlen=15)
        self.alo_probs = deque([0.99, 0.9], maxlen=15)
        self.bids_hist = deque([0.99, 0.9], maxlen=15)
        self.asks_hist = deque([0.99, 0.9], maxlen=15)
        self.mid = deque([], maxlen=5)
        self.tick = FIXTIC  # 1/64
        self.vwap = -1.0
        self.df = pd.DataFrame()
        self.filepath = './'
        # static const variables
        self.forcast_estimate_time = 90
        self.probfmt = '%2.9f'
        self.SAVEDEBUG = True
        self.buy_sum=deque([], maxlen=4)
        self.sell_sum=deque([], maxlen=4)
        logging.debug('{0} modelling manager initialised'.format(self.tradingpair))
        self.dic_probs = {}

        for hour in range(0,23):
            for minute in range(0,60,15):
                self.dic_probs[hour+minute/60]=(0,0)

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


    def fit_gamma(self, data):
        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(data)
        print(fit_alpha, fit_loc, fit_beta)
        return fit_alpha, fit_loc,fit_beta

    def loadfrompickle(self):
        file = open(self.filepath +'df/'+self.tradingpair+'dataframe.pkl', 'rb')
        tmp_dict = pickle.load(file)
        file.close()
        #self.__dict__.update(tmp_dict)
        return tmp_dict

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

        baspread = (self.asks[0,0] - self.bids[0,0])
        best=baspread
        if (x<baspread):
            best = x
        if y<baspread:
            best = y

        return best

    def get_sell_buy_order_rate(self, period_minutes=15):
        ut = dt.datetime.utcnow()
        numberofblocks = int(ut.minute/period_minutes)
        hour_fraction = ut.hour+((numberofblocks+1)*period_minutes/60)
        sell_order_per_sec, buy_order_per_sec = self.dic_probs.get(hour_fraction)
        return sell_order_per_sec, buy_order_per_sec

    def geo_mean_overflow(self, iterable):
        a = np.log(iterable)
        return np.exp(a.sum() / len(a))

    def prob_next_hour(self, sell_order_per_sec,buy_order_per_sec, timeframe, bins=5): # approximation based on linear
        sum_bids = self.bids[0:bins, :].sum(axis=0)
        sum_asks = self.asks[0:bins, :].sum(axis=0)
        #sell_order_per_sec, buy_order_per_sec = self.dic_probs.get(hour_fraction)
        if sell_order_per_sec is None:
            return -1
        if buy_order_per_sec is None:
            return -1
        #prob_blo_live = (sum_bids[1] - sell_order_per_sec)/ (sum_bids[1])
        #prob_alo_live = (sum_asks[1] - buy_order_per_sec) / (sum_asks[1])
        combined_prob = (sum_asks[1] + sum_bids[1] - buy_order_per_sec - sell_order_per_sec) / (sum_asks[1]+sum_bids[1])
        n = timeframe
        min_time_to_fill = 0

        n_prod = 1
        minprob =  combined_prob
        n_prod = np.power(minprob,timeframe)
        if minprob == 1:
            minprob==1-self.epsi
        min_time_to_fill = (1+(np.power(minprob, timeframe)*(timeframe-1)) - np.power(minprob, timeframe-1)*timeframe)/(1-minprob)
        return 1 - n_prod, min_time_to_fill*timeframe/3


    def probordercompletion2(self, timeframe, isask): # approximation based on linear
        n = timeframe
        if (isask and len(self.alo_probs)==0) or (len(self.blo_probs)==0 and not isask):
            return 0
        min_time_to_fill = 0
        if isask:
            n_prod = 1
            minprob =  np.min(self.alo_probs)
            minprobtime = np.mean(self.alo_probs)
            maxprobtime = np.max(self.alo_probs)
            if minprob == 1:
                minprob == 1 - self.epsi
            for n in self.alo_probs:
                n_prod = minprob* n_prod

            if timeframe>len(self.alo_probs):
                n_prod = np.power(n_prod,timeframe/len(self.alo_probs))
                # (np.power(minprobtime, timeframe)*(timeframe-1)) - np.power(minprobtime, timeframe-1)*timeframe)
                min_time_to_fill = 1/(1-minprobtime)
        else:
            n_prod = 1
            minprob = np.min(self.blo_probs)
            minprobtime = np.mean(self.blo_probs)
            maxprobtime = np.max(self.blo_probs)
            if minprob == 1:
                minprob == 1 - self.epsi
            for n in self.blo_probs:
                n_prod = minprob * n_prod
            if timeframe > len(self.blo_probs):
                n_prod = np.power(n_prod, timeframe / len(self.blo_probs))
                # (np.power(minprobtime, timeframe)*(timeframe-1)) - np.power(minprobtime, timeframe-1)*timeframe)
            min_time_to_fill = 1 / (1-minprobtime)
        return 1 - n_prod, [min_time_to_fill,1/(1-maxprobtime)]

    def probordercompletion(self, timeframe, isask): # approximation based on linear
        n = timeframe
        if (isask and len(self.alo_probs)==0) or (len(self.blo_probs)==0 and not isask):
            return 0
        if isask:
            return (1-np.power(self.geo_mean_overflow(self.alo_probs),n))  # here
        else:
            return (1-np.power(self.geo_mean_overflow(self.blo_probs),n))


    def set_lob_data(self, current_time, latest_ob ):
        self.bids = convert_to_ndarray(latest_ob['bids'], current_time)
        self.bids[:, 3] = np.zeros(len(self.bids))
        # savebids = self.bids
        # savebids = np.column_stack([self.bids,np.zeros(len(self.bids)) ])

        self.asks = convert_to_ndarray(latest_ob['asks'], current_time)
        self.asks[:, 3] = np.ones(len(self.asks))
    def getmarketorders_frombackupapi(self, mid):
        current_time = time.time()
        market_order_buy_sum = [-1, -1]
        market_order_sell_sum =[-1, -1]
        try:
            with urllib.request.urlopen(self.backup_tradeurl.format(self.tradingpair)) as url:
                latest_mo = json.loads(url.read().decode())
                print(latest_mo)
                # with open('marketorders.json', 'w') as outfile:
                #     json.dump(latest_mo, outfile)
                #if quoted['mid']>mid:
                #    mid = quoted['mid']
                temporders = np.array( [[  float(x['price']), float(x['amount']), int(x['date']) , 1 if x['price']<mid else 0]for x in latest_mo] )
                threshold = temporders[:,2]>int((current_time - self.tradewindow_sec-5) * 1000)
                filtorders = temporders[threshold]
                self.marketorders = filtorders
                logging.info('no of backup market orders'+str(len(self.marketorders)))
                print('no of backup market orders' + str(len(self.marketorders)))
                market_order_buy_sum = filtorders[(filtorders[:,-1]==0) & (filtorders[:,0]>self.bids[0, 0]), 0:2].sum(axis=0)
                market_order_sell_sum = filtorders[filtorders[:,-1]==1 & (filtorders[:,0]<self.asks[0, 0]), 0:2].sum(axis=0)
                return market_order_buy_sum[1], market_order_sell_sum[1]
        except urllib.error.HTTPError as detail:
            logging.info(self.tradingpair + ':')
            logging.exception( str(detail))
            if detail.errno in (401,500,404):
                print('exception http')
            return market_order_buy_sum[1], market_order_sell_sum[1]
        except ValueError as ver:
            logging.exception("ValueError"+str(ver))
            return market_order_buy_sum, market_order_sell_sum
        except Exception as eer:
            logging.exception("unexpected Error")
            logging.warning('Error, may not recover from this')
            return market_order_buy_sum, market_order_sell_sum
        except:
            logging.exception("Unhandled Exception")
            raise
        return market_order_buy_sum[1], market_order_sell_sum[1]

    def getlatestob(self,sfile, obfile, pfile):
        try:
            with urllib.request.urlopen(self.updateurl.format(self.tradingpair)) as url:
                current_time = dt.datetime.utcnow().timestamp()

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
                self.mid = (self.asks[0, 0] + self.bids[0, 0]) * 0.5
                (bvwap, avwap, totala, totalb) = calc_vwap(self.bids, self.asks)
                self.vwap = (bvwap*totalb+avwap*totala)/(totala+totalb)
                self.up_price = avwap
                self.down_price = bvwap
                # saveasks = self.asks

            with urllib.request.urlopen(self.tradeeurl.format(self.tradingpair)) as url:
                latest_mo = json.loads(url.read().decode())
                print(latest_mo)
                # with open('marketorders.json', 'w') as outfile:
                #     json.dump(latest_mo, outfile)
                temporders = np.array( [[  float(x['price']), float(x['qty']), int(x['time']) , 1 if x['isBuyerMaker'] else 0]for x in latest_mo] )
                threshold = temporders[:,2]>int((current_time - self.tradewindow_sec-5) * 1000)
                filtorders = temporders[threshold]
                buy_sum_back = 0
                sell_sum_back = 0
                if len(filtorders)<4:
                    buy_sum_back, sell_sum_back = self.getmarketorders_frombackupapi(self.mid)

                self.marketorders = filtorders
                print('no of market orders'+str(len(self.marketorders)))
                buy_sum = filtorders[filtorders[:,-1]==0].sum(axis=0)
                sell_sum = filtorders[filtorders[:,-1]==1].sum(axis=0)
                if len(buy_sum)>0:
                    buy_sum = buy_sum[1]
                else:
                    buy_sum = 0

                if len(sell_sum)>0:
                    sell_sum = sell_sum[1]
                else:
                    sell_sum = 0

                # if no trades on one side assume at least half as many as other side will arrive in near future based on back tests
                if buy_sum==0:
                    buy_sum = sell_sum*0.3
                if sell_sum==0:
                    sell_sum = buy_sum*0.3
                if buy_sum_back!=0 or sell_sum_back!=0:
                    buy_sum = np.max((buy_sum,buy_sum_back))
                    sell_sum = np.max((sell_sum, sell_sum_back))
                self.buy_sum.append(buy_sum)
                self.sell_sum.append(sell_sum)
                logging.info('no of market orders'+str(len(self.marketorders)))
                #market_order_buy_sum = filtorders[(filtorders[:,-1]==0) & (filtorders[:,0]>self.bids[0, 0]), 0:2].sum(axis=0)
                #market_order_sell_sum = filtorders[filtorders[:,-1]==1 & (filtorders[:,0]<self.asks[0, 0]), 0:2].sum(axis=0)
                buy_sum = np.mean(self.buy_sum)
                sell_sum = np.mean(self.sell_sum)

        except urllib.error.HTTPError as detail:
            logging.info('HTTPERROR'+ self.tradingpair)
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
            print(str(eer))
            return
        except:
            logging.info('very strange exception')
            logging.exception("Unhandled Exception")
            raise
        sum_bids = self.bids[0:5, :].sum(axis=0)
        sum_asks = self.asks[0:5, :].sum(axis=0)

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
        if self.tradingpair.upper()=='XBTUSD':
            if self.tick<0.5:
                self.tick=0.5
        mdf = pd.concat([dfbids, dfasks], ignore_index=True)
        try:
            nfs = convert_df_bins(mdf, self.bins)
            print(nfs)
        except Exception as e:
            print(e)
        self.df = nfs
        sell_order_per_sec = sell_sum/self.tradewindow_sec
        buy_order_per_sec = buy_sum/self.tradewindow_sec
        now = dt.datetime.utcnow()
        if (now.minute%15==0):
            self.dic_probs[now.hour+now.minute/60]=(sell_order_per_sec,buy_order_per_sec)
            logging.info(self.dic_probs)
        round_min = int(now.minute / 15) * 15 + 15
        round_min = now.hour + round_min/60
        self.stats.update(now.timestamp(), self.mid, self.tick*0.8)

        past_prob_blo_live = 0
        past_prob_alo_live = 0
        if round_min in self.dic_probs:
            past_order_per_sec = self.dic_probs[round_min]
            past_prob_blo_live = (sum_bids[1] - past_order_per_sec[0]) / (sum_bids[1])
            past_prob_alo_live = (sum_asks[1] - past_order_per_sec[1]) / (sum_asks[1])
            if(past_order_per_sec[0]>0):
                print('past order is greater than zero')
        prob_blo_live = (sum_bids[1]-sell_order_per_sec)/(sum_bids[1])
        prob_alo_live = (sum_asks[1] - buy_order_per_sec) / (sum_asks[1])
        floatfmt = '%30.9f'
        timefmt = '%30.3f'
        intfmt = '%i'
        mfmt = [floatfmt, floatfmt, timefmt, timefmt]
        if self.SAVEDEBUG:
            np.savetxt(sfile, filtorders, fmt="%30.10f",delimiter=',')
            np.savetxt(obfile, self.bids, fmt=mfmt, delimiter=',')
            np.savetxt(obfile, self.asks, fmt=mfmt, delimiter=',')
            bid_prob , btimetofill = self.probordercompletion2(self.forcast_estimate_time, 0)
            ask_prob, atimetofill = self.probordercompletion2(self.forcast_estimate_time, 1)
            bid_gmean = self.probordercompletion(self.forcast_estimate_time, 0)
            ask_gmean = self.probordercompletion(self.forcast_estimate_time, 1)
            thresh = 0.18
            self.price_prediction = self.mid
            if bid_prob>ask_prob+thresh:
                print('price falling')
                self.price_prediction = self.down_price
                self.stats.add_pred(current_time + 1500, self.price_prediction, self.price_prediction-self.mid)
            if ask_prob>bid_prob+thresh:
                print('price rising')
                self.price_prediction = self.up_price
                self.stats.add_pred(current_time + 1500, self.price_prediction, self.price_prediction-self.mid)
            self.confidence = abs(ask_prob-bid_prob)


            nn = np.array([[current_time * 1000, self.mid, past_prob_blo_live, prob_blo_live, prob_alo_live,
                            bid_prob,ask_prob,bid_gmean,ask_gmean, btimetofill[0], atimetofill[0],self.price_prediction]])
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
                (bvwap, avwap) = calc_vwap(self.bids, self.asks)
                self.vwap = (bvwap + avwap) / 2

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
                threshold = temporders[:,2]>int((current_time - self.tradewindow_sec-5) * 1000)
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
            bid_prob , btimetofill = self.probordercompletion2(self.forcast_estimate_time, 0)
            ask_prob, atimetofill = self.probordercompletion2(self.forcast_estimate_time, 1)
            nn = np.array([[current_time * 1000, mid,self.vwap, prob_blo_live, prob_alo_live, bid_prob, ask_prob ,self.probordercompletion(self.forcast_estimate_time ,0),  self.probordercompletion(self.forcast_estimate_time ,1)]])
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

    exchange = 'Binance'#'Bitmex'
    try:
        args = parser.parse_args()
        #print(args.validpair)
        name = args.validpair
        exchange = args.exchange
        if args.testmode=='yes':
            testmode = True
        if args.testmode == 'no' or args.testmode=='false':
            print('sorry currently do not support non testmode')
        if name is not None:
            print('tradingpair:' + str(name)+' exchange:'+str(exchange))
    except Exception as e:
        print("type error: " + str(e))
        name = 'EOSUSDT'
    if not testmode:
        ws = getBitmexWs()
    with mock.patch('bitmexwebsock.BitMEXWebsocket') as MockBitmexgetWs:
        ws = MockBitmexgetWs()


    if exchange is None:
        exchange = 'bitmexws'
        print('setting default exchange '+exchange)
    if name is not None:
        tp.name = name
    else:
        if exchange.lower() in ('bitmex', 'bitmexws'):
            tp.name = 'XBTUSD'  #'XBTUSD'
        else:
            tp.name = 'LTCUSDT' # for other exchanges

    tp.market = exchange
    if exchange.lower() == 'bitmex':
        cr = bitmexmanager(tp)
    else:
        cr = modellingmanager(tp)
        # cr.SAVEDEBUG = False
    prefix = './csv/'
    fp = open(prefix+str(tp.name)+'prob4.csv', 'ab')
    f = open(prefix+str(tp.name)+'marketorders4.csv', 'ab')
    fob = open(prefix+str(tp.name)+'orderbooks4.csv', 'ab')
    l = task.LoopingCall(cr.getlatestob,f,fob,fp)
    l.start(cr.tradewindow_sec)  # call every tradewindow_sec seconds

    # l.stop() will stop the looping calls
    reactor.run()
    # blo, slo = cr.getlatestob(f)
    logging.info('reactor completed')
    print('reactor completed')
    fp.close()
    f.close()
    fob.close()


