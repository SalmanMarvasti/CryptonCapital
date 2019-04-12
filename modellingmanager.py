import pandas as pd
import numpy as np
import urllib, json
import pickle# 5 as pickle
from collections import namedtuple, deque
import datetime as dt
import time
from twisted.internet import task
tp = namedtuple('tradingpair', ('name', 'datetime', 'market'))

from twisted.internet import reactor
import os
import time
import scipy.stats as stats
# from bitmex_websocket import getBitmexWs
from atomicwrites import atomic_write
from bitmexwebsock import getBitmexWs
import logging
from unittest import mock
from rsi import RSI



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

def calc_vwap(bids, asks, Rbins=8, mid=0):
    Rbins = min(asks.shape[0], min(bids.shape[0], Rbins))
    totalb = np.sum(bids[0:Rbins,1])
    totala = np.sum(asks[0:Rbins,1])
    bid_vwap=0
    ask_vwap=0
    for r in range(0, Rbins):
        bid_vwap += bids[r][0]*bids[r][1]
        ask_vwap += asks[r][0]*asks[r][1]
        if bid_vwap == 0:
            bid_vwap = mid
        if ask_vwap == 0:
            ask_vwap = mid

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
    FIXED_OFFSET = 1000
    def __init__(self, thresh=0.2, percent_cost = 0.0008):
        self.predlist = []
        self.number_of_predictions = 1 # prevent divide by zero erros
        self.number_correct = 0
        self.number_stopped = 0
        self.time_start = dt.datetime.utcnow().timestamp()
        self.time_end = 0
        self.prev_mid= 0
        self.tick = 0.5
        self.dollar_gain = 0
        #self.FIXED_OFFSET = 2000
        self.filledlist = []
        self.stoppedlist = []
        self.expiredlist = []
        self.thresh=thresh
        self.max_capital_deployed = 0
        self.cost = 0
        self.percent_cost = percent_cost

    def get_average_time_to_fill(self):
        total = 0
        if len(self.filledlist)==0:
            return (total, 0)
        confidence = 0
        for x in self.filledlist:
            total+=x[1]
            confidence+=x[2]
        return (total/len(self.filledlist), confidence/len(self.filledlist))

    def update(self, price, timestamp, tick):
        newlist = []
        if price>10000:
            raise Exception('invalid price')
        self.tick=tick
        for a in self.predlist:
            if timestamp<a[0]:
                if abs(price-a[1])<tick*0.5 or (a[2]>0 and price>a[1]) or (a[2]<0 and price<a[1]):
                    self.number_correct+=1
                    entry_price = a[6]
                    gain_amount = max(abs(a[2]),abs(price-entry_price))
                    self.dollar_gain+=gain_amount
                    self.filledlist.append((a, timestamp - a[0] + self.FIXED_OFFSET, gain_amount))
                else:
                    timepassed = timestamp-a[0]+self.FIXED_OFFSET
                    entry_price = a[6]
                    if a[2]<0: # gone short
                        price_loss = entry_price - price
                    else: # gone long
                        price_loss = price - entry_price
                    if ((price_loss)<-0.50 and abs(price_loss) > abs(price-a[4])) or (timepassed > 300 and (price_loss)<0 and abs(price_loss) > (0.25*abs(a[2])) ): # was 5 in last real time test
                        logging.info('order stopped')
                        loss_amount = abs(price_loss)
                        self.dollar_gain -= loss_amount
                        self.number_stopped += 1
                        self.stoppedlist.append((a,  timepassed, -1*loss_amount))
                        #self.cost+=
                    else: # keep order for now
                        if price_loss>1 and abs(price - a[4])>2: # we are making profit but not achieved target
                            a = (a[0], a[1], a[2], a[3], a[4] + price_loss*a[2]/abs(a[2]), a[5],a[6])

                        newlist.append(a)
            else:
                print('order expired')
                entry_price = a[1] - a[2]
                diff = a[2]
                if diff>0:
                    change_amount = price-entry_price
                    self.dollar_gain += change_amount
                else:
                    change_amount = entry_price - price
                    self.dollar_gain += change_amount
                self.expiredlist.append((a, change_amount))
                self.cost +=self.percent_cost*price
        self.predlist = newlist

    def __len__(self):
        return self.number_of_predictions
    def add_pred(self, validtill_timestamp, predicted_price, price_diff, confidence=0, sma=0, mid=0):
        if len(self.predlist)>0 and abs(predicted_price - self.predlist[-1][1])<(self.tick):
            print('ignoring duplicate prediction')
            return

        if len(self.predlist)>self.max_capital_deployed:
            self.max_capital_deployed = len(self.predlist)
        entry_price = mid
        if price_diff < 0:  # gone short
            stoploss = entry_price + 0.8*abs(price_diff)
        else:  # gone long
            stoploss = entry_price -  0.8*abs(price_diff)

        self.predlist.append((validtill_timestamp, predicted_price, price_diff, confidence, stoploss, sma, mid))
        self.number_of_predictions+=1
        if validtill_timestamp>self.time_end:
            self.time_end=validtill_timestamp
        self.cost+=0.0008 * (predicted_price+price_diff)


class modelob:
    def __init__(self, acct):
        global r
        self.tradingpair = acct.name
        self.backtest = False

        self.datetime = dt.datetime.utcnow()
        self.market = acct.market
        self.latestob = {}
        self.tradewindow_sec = 29
        self.epsi=0.00000000001
        self.stats = [prediction_checker(0.2), prediction_checker(0.3), prediction_checker(0.4), prediction_checker(0.5)]

        if self.market =='Binance':
            self.updateurl = "https://api.binance.com/api/v1/depth?symbol={0}"
            self.tradeeurl = "https://api.binance.com/api/v1/trades?symbol={0}"
            self.backup_tradeurl = "https://api.binance.com/api/v1/trades?symbol={0}"

        if self.market in ('bitmex', 'BitMex', 'Bitmex'): #'http://localhost:{port}/users'.format(port=53581)
            self.updateurl = "https://www.bitmex.com/api/bitcoincharts/{0}/orderBook"
            self.tradeeurl = "https://www.bitmex.com/api/bitcoincharts/{0}/trades"
            self.backup_tradeurl = "https://www.bitmex.com/api/bitcoincharts/{0}/trades"
            #
            self.tradewindow_sec = 25 # bitmex trade window must be lower as buyer or seller is not specified
        if self.market.lower() =='bitmexws':
            self.tradeeurl = 'http://localhost:{port}/users'.format(port=53581)
            self.updateurl = "https://www.bitmex.com/api/bitcoincharts/{0}/orderBook"
            self.backup_tradeurl = "https://www.bitmex.com/api/bitcoincharts/{0}/trades"
            self.tradewindow_sec = 24  #
        if self.market.lower() == 'backtest_bitmexws':
            self.tradeeurl = 'http://localhost:{port}/users'.format(port=65269)
            self.updateurl =  'http://localhost:{port}/users'.format(port=65110)
            self.backup_tradeurl = ""
            self.tradewindow_sec = 24  #
            self.backtest = True

        # .redis = connredis('redis.pinksphere.com')
        self.bins = []
        self.marketorders = []
        self.blo_probs = deque([0.99, 0.9], maxlen=12)
        self.alo_probs = deque([0.99, 0.9], maxlen=12)
        self.bids_hist = deque([0.99, 0.9], maxlen=15)
        self.asks_hist = deque([0.99, 0.9], maxlen=15)
        self.nfs_hist = deque(maxlen=10)
        self.mid_hist = deque([], maxlen=60)
        self.mid = 0
        self.tick = FIXTIC  # 1/64
        self.vwap = -1.0
        self.df = pd.DataFrame()
        self.filepath = './'    #'C:\\work\\Crypton\\CryptonCapital_alternative\\'
        # static const variables
        self.forcast_estimate_time = 90
        self.probfmt = '%2.9f'
        self.SAVEDEBUG = True
        self.buy_sum=deque([], maxlen=6)
        self.sell_sum=deque([], maxlen=6)
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
        self.bid_gmean = 0
        self.ask_gmean = 0
        self.orderarrival_bid = 0
        self.orderarrival_ask = 0
        self.orderadjustments_ask = deque([], maxlen=10)
        self.orderadjustments_bid = deque([], maxlen=10)
        self.up_pred_count = 0
        self.down_pred_count = 0
        self.last_pred_time = 0
        self.prev_current_time = 1
        self.adjust = False
        self.dollar_unit_cost = 3
        self.percent_cost = 0.001

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
# future stuff
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
            if minprob >= 1:
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
            if minprob >= 1:
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

    def predict_and_simtrade(self, current_time, bid_prob, ask_prob, prob_diff, x, up_price, down_price, thresh, sma):
        up = -1
        signal = 0
        self.stats[x].thresh = thresh
        timediff = current_time - self.last_pred_time
        if bid_prob > ask_prob + thresh:  # and (abs(diffg) > thresh*0.5 or abs(diffg)<0.05)
            print('price falling')
            up = 0
            price_prediction = down_price
            price_diff = price_prediction - self.mid

            if abs(price_diff) > 1.1:
                if timediff<1000:
                    self.down_pred_count += price_diff
                    logging.info('down_count' + str(self.down_pred_count))
                else:
                    print('reset downpredcount')
                    self.down_pred_count = 0
                netcount = self.up_pred_count + self.down_pred_count
                if (netcount <-4 and netcount > -30  or abs(prob_diff)>0.9)and (sma<self.mid ): #or abs(sma-self.mid)<self.tick
                    price_prediction = self.down_price
                    signal = -1
                    self.stats[x].add_pred(current_time + prediction_checker.FIXED_OFFSET, price_prediction-4,
                                  price_diff-4, prob_diff, sma, self.mid)
                self.last_pred_time = current_time

        if ask_prob > bid_prob + thresh:
            print('price rising')
            up = 1
            price_prediction = up_price
            price_diff = price_prediction - self.mid
            if abs(price_diff) > 1.1:
                if  timediff< 1000:
                    self.up_pred_count += price_diff
                    logging.info('up count '+str(self.up_pred_count)+' '+str(self.up_pred_count+self.down_pred_count))
                else:
                    self.up_pred_count = 0
                netcount = self.up_pred_count + self.down_pred_count
                if  (netcount> 4 and netcount<30 or abs(prob_diff)>0.9) and (sma>self.mid): # or abs(sma-self.mid)<self.tick
                    price_prediction = self.up_price
                    signal = 1
                    self.stats[x].add_pred(current_time + prediction_checker.FIXED_OFFSET, price_prediction+4,
                                       price_diff+4, prob_diff, sma, self.mid)
                self.last_pred_time = current_time
        if up==-1 and int(current_time)%prediction_checker.FIXED_OFFSET<self.tradewindow_sec:
            self.up_pred_count = 0
            self.down_pred_count = 0


        return up, signal

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
                return market_order_buy_sum[1], market_order_sell_sum[1], filtorders
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


    def findaskpos(self, nfs_df):
        pos = 5
        while nfs_df[pos,0]==0:
            pos = pos+1
        for i in range(int(pos), nfs_df.shape[0]):
            if nfs_df[i,0].left>0:  # found ask position
                return i

    def getlatestob(self,sfile, obfile, pfile):
        try:
            with urllib.request.urlopen(self.updateurl.format(self.tradingpair)) as url:

                current_time = 0
                latest_ob = json.loads(url.read().decode())
                # with open('limitorders.json', 'w') as outfile:
                #     json.dump(latest_ob, outfile)
                if self.backtest:
                    current_time = (float(latest_ob['current_time']))
                    if self.prev_current_time==1:
                        self.prev_current_time = current_time
                else:
                    current_time = dt.datetime.utcnow().timestamp()
                    self.prev_current_time = current_time - 1

                print(latest_ob)
                # CONVERTS DATA TO NDARRAY AND GIVES COL NAMES
                self.bids = convert_to_ndarray(latest_ob['bids'], current_time)

                self.asks = convert_to_ndarray(latest_ob['asks'], current_time, 1)
                if self.backtest:
                    self.bids[:,1]*=-1
                    min_size = max(self.bids.shape[0], 10)
                    if self.asks.shape[0]>10:
                        self.asks = self.asks[0:min_size,:]
                # self.asks[:,3] = np.ones(len(self.asks))
                self.mid = (self.asks[0, 0] + self.bids[0, 0]) * 0.5
                # self.mid_hist.append(self.mid)
                R = 6 # todo try 12
                (bvwap, avwap, totala, totalb) = calc_vwap(self.bids, self.asks,R, self.mid)
                self.vwap = (bvwap*totalb+avwap*totala)/(totala+totalb)
                self.up_price = avwap
                self.down_price = bvwap
                self.up_price_2 = avwap*1.001
                self.down_price_2 = bvwap*1.001

                # saveasks = self.asks

            with urllib.request.urlopen(self.tradeeurl.format(self.tradingpair)) as url:
                latest_mo = json.loads(url.read().decode())
                print(latest_mo)
                # with open('marketorders.json', 'w') as outfile:
                #     json.dump(latest_mo, outfile)
                temporders = np.array( [[  float(x['price']), float(x['qty']), int(x['time']) , 1 if x['isBuyerMaker'] else 0]for x in latest_mo] )
                threshold = temporders[:,2]>int((current_time - self.tradewindow_sec-3) * 1000)
                filtorders = temporders[threshold]
                buy_sum_back = 0
                sell_sum_back = 0
                print('no of market orders' + str(len(filtorders)))
                if not self.backtest and len(filtorders)<4:
                    buy_sum_back, sell_sum_back, filtorders = self.getmarketorders_frombackupapi(self.mid)

                self.marketorders = filtorders

                buy_marketorder_sum = filtorders[filtorders[:,-1]==0].sum(axis=0)
                sell_marketorder_sum = filtorders[filtorders[:,-1]==1].sum(axis=0)
                if len(buy_marketorder_sum)>0:
                    buy_marketorder_sum = buy_marketorder_sum[1]
                else:
                    buy_marketorder_sum = 0

                if len(sell_marketorder_sum)>0:
                    sell_marketorder_sum = sell_marketorder_sum[1]
                else:
                    sell_marketorder_sum = 0

                # if no trades on one side assume at least half as many as other side will arrive in near future based on back tests
                if buy_marketorder_sum==0:
                    buy_marketorder_sum = sell_marketorder_sum*0.3
                if sell_marketorder_sum==0:
                    sell_marketorder_sum = buy_marketorder_sum*0.3
                if buy_sum_back!=0 or sell_sum_back!=0:
                    buy_marketorder_sum = np.max((buy_marketorder_sum,buy_sum_back))
                    sell_marketorder_sum = np.max((sell_marketorder_sum, sell_sum_back))
                self.buy_sum.append(buy_marketorder_sum)
                self.sell_sum.append(sell_marketorder_sum)
                logging.info('no of market orders'+str(len(self.marketorders)))
                #market_order_buy_sum = filtorders[(filtorders[:,-1]==0) & (filtorders[:,0]>self.bids[0, 0]), 0:2].sum(axis=0)
                #market_order_sell_sum = filtorders[filtorders[:,-1]==1 & (filtorders[:,0]<self.asks[0, 0]), 0:2].sum(axis=0)
                buy_marketorder_sum = np.mean(self.buy_sum)
                sell_marketorder_sum = np.mean(self.sell_sum)

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
        # todo try uncomment

        if self.backtest:
            R=6
            if self.asks.shape[0]>20:
                R = 10
            self.asks = self.asks[np.append(self.asks[0:R, 1] > 0, ~np.isnan(self.asks[R:, 1]))]
            R = 6
            if self.bids.shape[0]>20:
                R = 10
            self.bids = self.bids[np.append(self.bids[0:R, 1] > 0, ~np.isnan(self.bids[R:, 1]))]
        max_R = min(min(self.asks.shape[0], 5), self.bids.shape[0])
        sum_range = range(0, max_R)
        sum_bids = self.bids[sum_range, :].sum(axis=0)
        sum_asks = self.asks[sum_range, :].sum(axis=0)
        bid_adjust = 0
        ask_adjust = 0
        if self.adjust:
            nfs_range = range(1, R-1)  # divide top by two
            if(len(self.orderadjustments_bid)>1):
                 bid_adjust = np.mean(self.orderadjustments_bid, axis=0)[nfs_range].sum()
                 ask_adjust = np.mean(self.orderadjustments_ask, axis=0)[nfs_range].sum()

        orderbook_pricelevels = np.concatenate((self.bids[:,0], self.asks[:,0]))
        self.bins = np.histogram_bin_edges(orderbook_pricelevels, bins='fd')
        dfbids = pd.DataFrame({'price':self.bids[:,0] - self.mid, 'amount':self.bids[:,1],})
        dfasks = pd.DataFrame({'price': self.asks[:, 0]- self.mid, 'amount': self.asks[:, 1], })
        tick = self.bins[1]-self.bins[0]
        self.bins = np.histogram_bin_edges(orderbook_pricelevels, bins='doane')
        tick_do = self.bins[1] - self.bins[0]
        now = dt.datetime.utcnow()

        timepassed = (now.timestamp() - self.datetime.timestamp())
        if(timepassed%240<10):
            logging.info('mid added')
            self.mid_hist.append(self.mid)
        if self.backtest:
            timepassed *= (current_time-self.prev_current_time)
            if timepassed!=0:
                print ('time each frame '+ str(self.tradewindow_sec))
                self.tradewindow_sec = (current_time-self.prev_current_time)
                self.forcast_estimate_time = self.tradewindow_sec*4


        if timepassed<100:
            self.tick = self.choosetick(tick_do/2, tick/3)

        for st in self.stats:
            st.update(self.mid,current_time, self.tick)
        if (self.backtest):
            bin_tick = 0.5
        else:
            bin_tick = 2*self.tick
        self.bins = np.arange(dfbids['price'].min(), dfasks['price'].max(), bin_tick)
        # if self.tradingpair.upper()=='XBTUSD':
        #     if self.tick<0.5:
        #         self.tick=0.5
        mdf = pd.concat([dfbids, dfasks], ignore_index=True)


        try:
            nfs_df = convert_df_bins(mdf, self.bins)
            nfs_df.index.name = 'priceindex'
            print(nfs_df)
            R = 4
            if self.bid_gmean>0.3 or self.ask_gmean>0.3:
                R = 6
            nfs_np = nfs_df.reset_index().values

            if self.backtest:
                if nfs_np.shape[0]>40:
                    pad_value = int(np.ceil((nfs_np.shape[0]-40 )/ 2))
                    nfs_np = nfs_np[pad_value:-1*pad_value, :]
                pad_value = int(np.ceil((40 - nfs_np.shape[0]) / 2))
                if pad_value>0:
                    nfs_np = np.pad(nfs_np, ((pad_value, pad_value), (0, 0)), 'constant')
                nfs_np = nfs_np[0:40, :]

            if ( timepassed > 100):
                if nfs_np.shape[0] > self.nfs_hist[-1].shape[0]:
                    row_remove = nfs_np.shape[0] - self.nfs_hist[-1].shape[0]
                    nfs_np = nfs_np[:-row_remove,:]
                    logging.info('removing ' + str(row_remove) + ' from NFS')


                net_rate = nfs_np[:,1:] - self.nfs_hist[-1]
                net_rate = net_rate[:,1]
                orderarrival_ask = self.orderarrival_ask
                orderarrival_bid = self.orderarrival_bid

                avg_buy_sum = (self.buy_sum[-1] + self.buy_sum[-2])*0.5
                avg_sell_sum = (self.sell_sum[-1] + self.sell_sum[-2])*0.5
                if self.backtest:
                    ask_pos  = self.findaskpos(nfs_np)
                else:
                    ask_pos = 12

                if buy_marketorder_sum < self.nfs_hist[-1][ask_pos,1]:
                    orderarrival_ask = net_rate[ask_pos] + avg_buy_sum
                if sell_marketorder_sum < self.nfs_hist[-1][ask_pos-1,1]:
                    orderarrival_bid = net_rate[ask_pos-1] + avg_sell_sum

                self.orderadjustments_ask.append(net_rate[ask_pos:ask_pos+10])
                self.orderadjustments_bid.append(net_rate[0:ask_pos][::-1][0:10])
                self.orderarrival_ask = orderarrival_ask
                self.orderarrival_bid = orderarrival_bid

                mean_np = np.mean(self.nfs_hist,axis=0)
                mid_level = 4
                mean_np[ask_pos-mid_level:ask_pos+mid_level]=nfs_df.values[ask_pos-mid_level:ask_pos+mid_level]
                mean_nfs = pd.DataFrame.from_records(mean_np, columns=['price', 'amount'])
                bid,ask, totalask, totalbid = calc_vwap(mean_nfs.loc[mean_nfs.price<0].sort_index(0, ascending=False).values, mean_nfs.loc[mean_nfs.price>0].values, R, 0)
                self.down_price_2 = self.mid + bid*0.50
                self.up_price_2 = self.mid + ask*0.50

        except Exception as e:
            print('Exception NFS' + str(e))
        self.df = nfs_df
        sell_order_per_sec = sell_marketorder_sum/self.tradewindow_sec
        buy_order_per_sec = buy_marketorder_sum/self.tradewindow_sec



        if (now.minute%15==0):
            self.dic_probs[now.hour+now.minute/60]=(sell_order_per_sec,buy_order_per_sec)
            logging.info(self.dic_probs)
        round_min = int(now.minute / 15) * 15 + 15
        round_min = now.hour + round_min/60


        past_prob_blo_live = 0
        past_prob_alo_live = 0
        if timepassed> 3600*24 and round_min in self.dic_probs:
            past_order_per_sec = self.dic_probs[round_min]
            past_prob_blo_live = (sum_bids[1] - past_order_per_sec[0]) / (sum_bids[1])
            past_prob_alo_live = (sum_asks[1] - past_order_per_sec[1]) / (sum_asks[1])
            if(past_order_per_sec[0]>0):
                print('past order is greater than zero')
        self.orderarrival_bid /= self.tradewindow_sec
        self.orderarrival_ask /= self.tradewindow_sec
        if self.adjust:
            bid_adjust /= self.tradewindow_sec
            ask_adjust /= self.tradewindow_sec

            if(sum_bids[1] + bid_adjust < 0):
                 bid_adjust = 0
            if(sum_asks[1]+ask_adjust<0):
                 ask_adjust = 0
        if(sum_bids[1] + self.orderarrival_bid)<0:
            logging.info('Warning prob could be negative')
            self.orderarrival_bid = 0
        if sum_asks[1] + self.orderarrival_ask<0:
            logging.info('Warning asks prob could be negative')
            self.orderarrival_ask = 0

        prob_blo_live = (sum_bids[1] + bid_adjust + self.orderarrival_bid - sell_order_per_sec)/(bid_adjust+sum_bids[1] + self.orderarrival_bid)
        prob_alo_live = (sum_asks[1] + ask_adjust + self.orderarrival_ask - buy_order_per_sec) / (ask_adjust+sum_asks[1] + self.orderarrival_ask)
        try:
            if prob_blo_live<0 or prob_alo_live<0:
                prob_blo_live = 0.99
                prob_alo_live= 0.99
                if len(self.orderadjustments_bid)>3:
                    bid_adjust = np.mean(self.orderadjustments_bid, axis=0)[1:4].sum()
                    ask_adjust = np.mean(self.orderadjustments_ask, axis=0)[1:4].sum()
                    prob_blo_live = (sum_bids[1] + bid_adjust + self.orderarrival_bid - sell_order_per_sec) / (
                                bid_adjust + sum_bids[1] + self.orderarrival_bid)
                    prob_alo_live = (sum_asks[1] + ask_adjust + self.orderarrival_ask - buy_order_per_sec) / (
                                ask_adjust + sum_asks[1] + self.orderarrival_ask)
                    if prob_blo_live < 0 or prob_alo_live < 0:
                        prob_alo_live = np.mean(self.alo_probs)
                        prob_blo_live = np.mean(self.blo_probs)
        except Exception as e:
            logging.exception('prob_blo NEGATIVE:'+str(e))


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
            if self.bid_gmean > 0.4 or self.ask_gmean > 0.4:
                R = 7 # should be 7 for this version
                bvwap, avwap, totala, totalb = calc_vwap(self.bids, self.asks, R, self.mid)
                self.up_price = avwap
                self.down_price = bvwap
            thresh = 0.19
            self.price_prediction = self.mid
            diffg = bid_prob - ask_prob

            signal2 = 0
            signal1 = 0
            sma = np.mean(self.mid_hist)
            up, signal1 = self.predict_and_simtrade(current_time, bid_prob, ask_prob, diffg, 0,self.up_price, self.down_price, thresh, sma)

            if(up==1):
                self.price_prediction = self.up_price + 4
            else:
                if up==0:
                    self.price_prediction = self.down_price - 4


            self.confidence = abs(ask_prob - bid_prob)
            thresh=0.20
            if ask_gmean>0.1 and bid_gmean>0.1 and self.confidence>0.19:
                up, signal2 = self.predict_and_simtrade(current_time, bid_gmean, ask_gmean, diffg, 3, self.up_price_2, self.down_price_2,thresh, sma)
                if(up==1):
                    self.price_prediction = self.up_price_2 + 4
                else:
                    if up==0:
                        self.price_prediction = self.down_price_2 - 4

            rsi_ind =  (RSI(pd.Series(self.mid_hist), 7))[-1]
            nn = np.array([[current_time * 1000, self.mid, prob_blo_live, prob_alo_live,
                            bid_prob,ask_prob,bid_gmean,ask_gmean, btimetofill[0], atimetofill[0],self.price_prediction, self.up_pred_count+self.down_pred_count, sma, rsi_ind , signal2]])
            nnfmt = self.get_fmt_list(timefmt, self.probfmt, nn.shape[1])
            np.savetxt(pfile, nn, fmt=nnfmt, delimiter=',')
            pfile.flush()
            sfile.flush()
        self.dollar_unit_cost = (sma * self.percent_cost)*0.75
        self.bid_gmean = bid_gmean
        self.ask_gmean = ask_gmean
        if not self.backtest or nfs_np.shape[0]==40:
            self.nfs_hist.append(nfs_np[:,1:])
        self.bids_hist.append(self.bids)
        self.asks_hist.appendleft(self.asks)
        self.alo_probs.append(prob_alo_live)
        self.blo_probs.append(prob_blo_live)

        self.prev_current_time = current_time
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
                (bvwap, avwap) = calc_vwap(self.bids, self.asks, 6, self.mid)
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
        if exchange.lower() in ('bitmex', 'bitmexws', 'backtest_bitmexws'):
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
    if exchange.lower().find('backtest')>-1:
        l.start(4)
    else:
        l.start(cr.tradewindow_sec-3)  # call every tradewindow_sec seconds

    # l.stop() will stop the looping calls
    reactor.run()
    # blo, slo = cr.getlatestob(f)
    logging.info('reactor completed')
    print('reactor completed')
    fp.close()
    f.close()
    fob.close()


