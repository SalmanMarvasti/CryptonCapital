import redis
import time
import json
from collections import namedtuple
import threading
import logging
import random
import queue
from modellingmanager import create_model, modellingmanager, bitmexmanager
import numpy as np
import datetime
credentials = namedtuple('credentials', ('api', 'secret', 'endpoint'))
r = redis.StrictRedis(host='localhost',  port=6379, db=0)
max_retries = 10



logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',
                    filename='./responsemanager.log',
                    filemode='w'
                    )

BUF_SIZE = 100
q = queue.Queue(BUF_SIZE)


class RequestThread(threading.Thread):
    def __init__(self, group=None, target='test', name=None,
                 args=(), kwargs=None, verbose=None):
        super(RequestThread, self).__init__()
        self.target = target
        self.name = name

    def run(self):
        while True:
            if not q.full():
                credentials.api = "1f28b5ccdec84a30bbd0231bb210c7d7"
                credentials.secret = "Test@123"
                credentials.endpoint = "redis.pinksphere.com"
                cr = PublishServer(credentials)
                cr.read(self.target)

                time.sleep(random.random())
        return


class ResponseThread(threading.Thread):
    def __init__(self, group=None, target='test', name=None,
                 args=(), kwargs=None, verbose=None):
        super(ResponseThread, self).__init__()
        self.target = target
        self.name = name
        return

    def run(self):
        while True:
            if not q.empty():

                logging.debug('Running PublishServer class'
                              + ' : ' + str(q.qsize()) + ' items in redis')
                credentials.api = "1f28b5ccdec84a30bbd0231bb210c7d7"
                credentials.secret = "Test@123"
                credentials.endpoint = "redis.pinksphere.com"
                cr = PublishServer(credentials)
                cr.publish(self.target)

                time.sleep(random.random())
        time.sleep(1)
        return




def try_command(f, *args, **kwargs):
    global r
    count = 0
    while True:
        try:
            return f(*args, **kwargs)
        except redis.ConnectionError:
            count += 1

            # re-raise the ConnectionError if we've exceeded max_retries
            if count > max_retries:
                raise

            backoff = count * 2

            print('Retrying in {} seconds'.format(backoff))
            time.sleep(backoff)
            r = connredis('redis.pinksphere.com')


def connredis(h):
    r = redis.StrictRedis(host='ab722e68624e211e9b8160e2a8d9724d-949947629.us-east-1.elb.amazonaws.com',password='Test@123', port=6379, db=0)
    # redis.StrictRedis(host=h, password='Test@123', port=6379, db=0)
    return r

class PublishServer:
    """
        This class implements coinigy's REST api as documented in the documentation
        available at
        https://github.com/coinigy/api
    """

    def __init__(self, acct):
        global r
        self.api = acct.api
        self.secret = acct.secret
        self.endpoint = acct.endpoint
        self.redis = connredis('redis.pinksphere.com')
        self.mosize = 0
        r = self.redis

    def readpickleddataframe(self, tpair, exch='Binance'):
        return create_model(tpair, exch) # currently loads from file


    def publish(self, chann='test', isjson=False, **args):
        global r
        channel = r.pubsub()
        i = 0
        while True:


            # tradetype = 'buy'
            item = q.get()
            logging.debug('response item'+str(item))
            jl = json.loads(item)
            ts = float(jl['tradesize'])
            tradearray = []
            rd = random.random()

            tradetype = jl['type']
            tradetype = tradetype.lower()
            o = self.readpickleddataframe(jl['pair'])
            mid = (o.asks[0, 0] + o.bids[0, 0]) * 0.5
            scale = 0.8
            askvol = 0
            bidvol = 0
            tickd = 0


            num_bins_used = 4

            for i in range(0,num_bins_used):
                askvol += o.asks[i, 1] * scale
                bidvol += o.bids[i, 1] * scale
            noimpact_vol=1
            mosize = np.sum(o.marketorders[:,1])/len(o.marketorders)
            self.mosize=mosize*0.7+self.mosize*0.3
            mosize = self.mosize
            buy_int = -1
            first_queue_length = bidvol
            if tradetype.lower() == 'buy':
                first_queue_length = o.vol_at_lob(1, 1)
                noimpact_vol = askvol/num_bins_used*0.009
                buy_int = 1
            else:
                first_queue_length = o.vol_at_lob(1, 1)
                noimpact_vol = bidvol/num_bins_used*0.009
                buy_int = -1
            if mosize<noimpact_vol:
                noimpact_vol = mosize*0.95

            prob_order_fill = o.probordercompletion(int(jl['time_seconds']),tradetype=='buy')
            alt_prob_order_fill =  o.probordercompletion2(int(jl['time_seconds']),tradetype=='buy')
            marketorderint = 0
            if prob_order_fill>0.1 or alt_prob_order_fill>0.15:
                marketorderint = int(10*prob_order_fill)
                if marketorderint>3:
                    marketorderint = int(marketorderint/2)

            if ts<noimpact_vol:
                tradearray.append((0.5+(rd*0.1)) * ts)
                tradearray.append((0.5 - (rd * 0.1)) * ts)
                #tradearray.append((0.3) * ts)
                if mosize>first_queue_length/10:
                    tickd = -1*(first_queue_length/mosize)/10*num_bins_used
                    if tickd > -1:
                        tickd = -1
                if mosize * 0.5 > noimpact_vol:
                    tickd = -1
                ticksaway = [0, buy_int*tickd]
            else:
                number_trades = int((ts/noimpact_vol)*num_bins_used)

                if number_trades<2:
                    number_trades = 2
                ticksaway = []
                for j in range(0, int(number_trades)):
                    tradearray.append(rd * ts/number_trades)
                    tradearray.append((1-rd) * ts/number_trades)
                    if j>number_trades*(alt_prob_order_fill+(0.1*(0.5-float(jl['targetcost_percent'])))):
                        marketorderint=0
                    ticksaway.append(buy_int*-1*marketorderint)
                    ticksaway.append(buy_int * -1*marketorderint)
                    rd = 0.5+random.random()/4


            mydict = {'id': jl['id'], 'valid_for_sec': o.forcast_estimate_time*4, 'timestamp': datetime.datetime.utcnow().timestamp(), 'no_blocks': len(tradearray), 'ticksize': 0.50, 'pair': jl['pair'], 'trade_size': tradearray, 'type': tradetype, 'price': ticksaway[:len(tradearray)], 'prob_fill': prob_order_fill, 'alt_prob': alt_prob_order_fill }
            print('publishing'+str(mydict))
            rval = json.dumps(mydict)
            try_command(r.publish, chann, rval)
            print('done publish')
            time.sleep(0.5)
            i = i + 1




    def read(self, chan='test'):
        global r
        p = r.pubsub()
        p.subscribe(chan)
        c = 0

        while c<10:

            message = try_command(p.get_message)
            # message = p.get_message()
            if message and message['type']=='message':
                print('received publish and getting message')
                item = message['data']
                print( "Subscriber: %s" % item)
                q.put(str(message['data'].decode('utf-8')))
                logging.debug('Putting ' + str(item)
                              + ' : ' + str(q.qsize()) + ' items in redis')

            #time.sleep(1)





if __name__ == "__main__":
    pfill = 0.7
    tradearray = [5]
    tradetype = 'buy'
    #mydict = {'id': random.randint(1,1000), 'pair':'EOSUSDT','no_blocks': 3, 'trade_size': tradearray, 'type': tradetype, 'price': [-1, -2, -3],
    #           'prob_fill': pfill, 'ticksize':0.01}
    #
    # q.put(json.dumps(mydict))
    # for i in range(1, 10):
    #     mydict['id'] = random.random()
    #     q.put(json.dumps(mydict))
    # mydict = {'id': random.randint(1, 1000), 'pair': 'XBTUSD', 'type': tradetype, 'targetcost_percent': 0.1,
    #          'exchange': 'Bitmex', 'tradesize': 3200, 'time_seconds': 120}

    # q.put(json.dumps(mydict))

    p = RequestThread(name='request',target='trade')
    c = ResponseThread(name='response',target='traderesponse')

    c.start()
    time.sleep(2)
    p.start()
    time.sleep(2)
    c.join()
    p.join()

