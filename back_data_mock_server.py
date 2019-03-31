# Standard library imports...
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
from threading import Thread
import io
import json
# Third-party imports...
# from nose.tools import assert_true
import pickle
import requests
import pandas as pd
import numpy as np
import datetime as dt
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

def remove_duplicat_price_bidask(nob):
    aaa = nob.reset_index()
    aaa.loc[aaa.type == 1, ['amount']] = aaa[aaa.type == 1].amount * -1
    bbb = aaa.groupby(['price', 'date']).sum()
    return bbb.reset_index()

class MockServerRequestHandler(BaseHTTPRequestHandler):
    time_ind = 0
    year=2018
    month=12
    day=1
    exchange='BITMEX'
    pair_ix='PERP_BTC_USD'
    base_dir='./ob/'

    def genFileName(self,ind):
        str(dt.datetime(day=self.day, month=self.month, year=self.year))[:10] +self.exchange+'-'+self.pair_ix+str(ind)


    def get_lob(self):
        filestr = (self.genFileName(self.time_ind))
        lobfile = self.base_dir+filestr
        print(filestr)
        f = open(lobfile, 'rb')
        lob = pickle.load(f)
        return lob
        # trades = pickle

    def get_trades(self):
        filestr = (self.genFileName(self.time_ind))
        tradefile = self.base_dir+filestr
        #tradefile =
        print(filestr)
        f = open(tradefile, 'rb')
        return pickle.load(filestr)
        # trades = pickle


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


    # for lob    dic = {'pair': pair, 'price': tradedf.loc[x,'price'], 'qty':tradedf.loc[x,'amount'],'time': tradedf.loc[x,'date'], 'isBuyerMaker': )}
    #
    def createListFromTradesDF(self, tradedf):
        #arrayTrades = np.zeros(len(listTrades), 4)
        global pair
        blist = []
        r=0
        for x in range(0 , tradedf.shape[0]):
            dic = {'pair': pair, 'price': tradedf.loc[x,'price'], 'qty' : tradedf.loc[x,'base_amount'] , 'time': tradedf.loc[x , 'time_exchange'], 'isBuyerMaker': tradedf.loc[x , 'taker_side']}
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
        body = self.recent_trades()
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