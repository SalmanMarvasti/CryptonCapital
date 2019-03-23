# Standard library imports...
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
from threading import Thread
import io
import json
# Third-party imports...
# from nose.tools import assert_true
import requests
#import numpy as np
from bitmexwebsock import getBitmexWs
import ciso8601
def convert_utc_to_epoch_trades(timestamp_string):
    '''Use this function to convert utc to epoch'''
    #temp = timestamp_string.split('.')
    #con = temp[0]
    #timestamp = datetime.strptime(con, '%Y-%m-%dT%H:%M:%S')
    timestamp = ciso8601.parse_datetime(timestamp_string)
    epoch = timestamp.timestamp()
    return epoch*1000


class MockServerRequestHandler(BaseHTTPRequestHandler):

    def createListFromTrades(self, listTrades):
        #arrayTrades = np.zeros(len(listTrades), 4)
        global pair
        blist = []
        r=0
        for trade in listTrades:
            #arrayTrades[r, trade.]
            dic = {'pair': pair, 'price': trade['price'], 'qty':trade['size'],'time': convert_utc_to_epoch_trades(trade['timestamp']), 'isBuyerMaker': int(trade['side'].lower()!='buy')}
            blist.append(dic)
            r = r+1
        return blist


    def do_GET(self):
        # Process an HTTP GET request and return a response with an HTTP 200 status.
        self.send_response(requests.codes.ok, "thanks for contacting us")
        self.end_headers()

        #response = io.BytesIO()
        #response.write(b'This is POST request. ')
        #response.write(b'Received: ')
        body = ws.recent_trades()
        ltrades = self.createListFromTrades(body)
        s = json.dumps(ltrades)
        if len(ltrades) != 0:
            print(ltrades[-1])

        #response.write(s)

        self.wfile.write(s.encode('utf-8'))

        return


def get_free_port():
    s = socket.socket(socket.AF_INET, type=socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    address, port = s.getsockname()
    s.close()
    print('using port: '+str(port))
    return port


class TestMockServer(object):

    def __init__(self, name):
        self.tradepair = name
    @classmethod
    def setup_class(cls):
        # Configure mock server.
        cls.mock_server_port = 53581 #get_free_port()
        cls.mock_server = HTTPServer(('localhost', cls.mock_server_port), MockServerRequestHandler)

        # Start running mock server in a separate thread.
        # Daemon threads automatically shut down when the main process exits.
        cls.mock_server_thread = Thread(target=cls.mock_server.serve_forever)
        cls.mock_server_thread.setDaemon(True)
        cls.mock_server_thread.start()

    def test_request_response(self):
        url = 'http://localhost:{port}/users'.format(port=self.mock_server_port)

        # Send a request to the mock API server and store the response.
        response = requests.get(url)

        # Confirm that the request-response cycle completed successfully.
        print(response.reason)
        jtrades = json.loads(response.content.decode('utf-8'))
        print(jtrades)
        #assert_true(response.ok)



import time
import sys
if __name__ == "__main__":
    global pair
    pair = 'XBTUSD'
    if len(sys.argv)>1:
        pair = sys.argv[1]
        ws = getBitmexWs(sys.argv[1])
    else:
        ws = getBitmexWs(pair)
    t=TestMockServer(pair)
    t.setup_class()
    t.test_request_response()
    while True:
        time.sleep(1)



pair = ''