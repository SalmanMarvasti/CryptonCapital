import redis
import time
from collections import namedtuple
credentials = namedtuple('credentials', ('api', 'secret', 'endpoint'))

class PublishServer:
    """
        This class implements coinigy's REST api as documented in the documentation
        available at
        https://github.com/coinigy/api
    """

    def __init__(self, acct):
        self.api = acct.api
        self.secret = acct.secret
        self.endpoint = acct.endpoint
        self.queue = redis.StrictRedis(host='redis.pinksphere.com', password='Test@123', port=6379, db=0)

    def publish(self, query=None, json=False, **args):
        channel = self.queue.pubsub()
        for i in range(10):
            print('published'+str(i))
            self.queue.publish("test", i)
            time.sleep(0.5)



    def read(self):
        r = self.queue
        p = r.pubsub()
        p.subscribe('test')

        while True:
            print('get_message')
            message = p.get_message()
            if message:
                print( "Subscriber: %s" % message['data'])
            time.sleep(1)





if __name__ == "__main__":

    credentials.api = "1f28b5ccdec84a30bbd0231bb210c7d7"
    credentials.secret = "Test@123"
    credentials.endpoint = "https://api.coinigy.com/api/v2"
    cr = PublishServer(credentials)
    cr.publish()

