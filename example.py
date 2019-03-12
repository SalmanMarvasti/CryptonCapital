import redis
import time
import json
from collections import namedtuple
import threading
import logging
import random
import queue
# from modellingmanager import create_model, modellingmanager, bitmexmanager
credentials = namedtuple('credentials', ('api', 'secret', 'endpoint'))
r  = redis.StrictRedis(host='localhost',  port=6379, db=0)
max_retries = 10



logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s', )

BUF_SIZE = 100
q = queue.Queue(BUF_SIZE)
qres =  queue.Queue(BUF_SIZE)

class ReadRequestResponseThread(threading.Thread):
    def __init__(self, group=None, target='test', name=None,
                 args=(), kwargs=None, verbose=None):
        super(ReadRequestResponseThread, self).__init__()
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


class RequestThread(threading.Thread):
    def __init__(self, group=None, target='test', name=None,
                 args=(), kwargs=None, verbose=None):
        super(RequestThread, self).__init__()
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
                cr.requestItems(self.target)

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
        r = self.redis


    def requestItems(self, chann='test', isjson=False, **args):
        global r
        channel = r.pubsub()
        i = 0
        while True:

            pfill = 0.7

            tradetype = 'buy'
            item = q.get()
            print(str(item))
            logging.debug('requesting item'+str(item))
            jl = json.loads(item)


            rval = json.dumps(mydict)
            try_command(r.publish,chann, rval)
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
                print('received requestItems and getting message')
                item = message['data']
                print( "Subscriber: %s" % item)
                qres.put(str(message['data'].decode('utf-8')))
                logging.debug('Putting ' + str(item)
                              + ' : ' + str(q.qsize()) + ' items in redis')

            #time.sleep(1)





if __name__ == "__main__":
    pfill = 0.7
    tradearray = [5]
    tradetype = 'buy'
    mydict = {'id': random.randint(1,1000), 'pair':'XBTUSD', 'type': tradetype, 'targetcost_percent': 0.1, 'exchange':'Bitmex', 'tradesize':10, 'time_seconds': 120}

    q.put(json.dumps(mydict))
    for i in range(1, 4):
        mydict['id'] = random.random()
        q.put(json.dumps(mydict))
        mydict['type'] = 'sell'
    p = RequestThread(name='request', target='trade')
    c = ReadRequestResponseThread(name='response', target='traderesponse' )

    p.start()
    #time.sleep(7)
    c.start()
    print('adding more stuff to queue')
    for i in range(1, 10):
        mydict['id'] = random.random()

    p.join()
    c.join()


