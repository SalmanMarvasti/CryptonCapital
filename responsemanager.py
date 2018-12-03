import redis
import time
import json
from collections import namedtuple
import threading
import logging
import random
import queue
credentials = namedtuple('credentials', ('api', 'secret', 'endpoint'))
r = redis.StrictRedis(host='localhost',  port=6379, db=0)
max_retries = 10



logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s', )

BUF_SIZE = 100
q = queue.Queue(BUF_SIZE)


class RequestThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
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
                cr.read('tradesizerequest')

                time.sleep(random.random())
        return


class ResponseThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(ResponseThread, self).__init__()
        self.target = target
        self.name = name
        return

    def run(self):
        while True:
            if not q.empty():

                logging.debug('Running PublishServer class'
                              + ' : ' + str(q.qsize()) + ' items in queue')
                credentials.api = "1f28b5ccdec84a30bbd0231bb210c7d7"
                credentials.secret = "Test@123"
                credentials.endpoint = "redis.pinksphere.com"
                cr = PublishServer(credentials)
                cr.publish('tradesizeresponse')

                time.sleep(random.random())
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
    r = redis.StrictRedis(host=h, password='Test@123', port=6379, db=0)
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
        self.queue = connredis('redis.pinksphere.com')
        r = self.queue

    def publish(self, query='test', isjson=False, **args):
        global r
        channel = r.pubsub()
        i = 0
        while True:
            print('published'+str(i))
            pfill = 0.7
            tradearray = [5]
            tradetype = 'buy'
            item = q.get()
            jl = json.loads(item)
            mydict = {'id': jl['id'], 'no_blocks': 3, 'trade_size': tradearray, 'type': tradetype, 'price': [-1, -2 , -3], 'prob_fill': pfill }
            rval = json.dumps(mydict)
            try_command(r.publish,query, rval)
            time.sleep(0.5)
            i = i + 1




    def read(self, chan='test'):
        global r
        p = r.pubsub()
        p.subscribe(chan)
        c = 0

        while c<10:
            print('get_message')
            message = try_command(p.get_message)
            # message = p.get_message()
            if message:
                item = message['data']
                print( "Subscriber: %s" % item)
                q.put(message['data'])
                logging.debug('Putting ' + str(item)
                              + ' : ' + str(q.qsize()) + ' items in queue')

            time.sleep(1)





if __name__ == "__main__":
    pfill = 0.7
    tradearray = [5]
    tradetype = 'buy'
    mydict = {'id': random.randint(1,1000), 'no_blocks': 3, 'trade_size': tradearray, 'type': tradetype, 'price': [-1, -2, -3],
              'prob_fill': pfill, 'ticksize':0.01}

    q.put(json.dumps(mydict))
    for i in range(1, 10):
        mydict['id'] = random.random()
        q.put(json.dumps(mydict))
    # p = ProducerThread(name='producer')
    c = ResponseThread(name='response')

    c.start()
    time.sleep(2)
    # c.start()
    time.sleep(2)


