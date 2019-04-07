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

time_ind = 0

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
    bbb = bbb.reset_index()
    bbb.loc[bbb.type == 1, ['amount']] = bbb[bbb.type == 1].amount * -1
    return bbb

class MockServerRequestHandler(BaseHTTPRequestHandler):

    year=2018
    month=12
    day=3
    exchange='BITMEX'
    pair_ix='PERP_BTC_USD'
    base_dir='./ob/'

    # def __init__(self):
    #     global pair
    #     if pair.upper()!='XBTUSD':
    #         exchange='BINANCE'

    def genFileName(self,ind):
        return str(dt.datetime(day=self.day, month=self.month, year=self.year))[:10] + ' ' +self.exchange+'_'+self.pair_ix+str(ind)


    def get_lob(self):
        global time_ind
        filestr = (self.genFileName(time_ind))
        lobfile = self.base_dir+filestr
        print(filestr)
        f = open(lobfile, 'rb')
        lob = pickle.load(f)
        return lob.reset_index()
        # trades = pickle

    def get_trades(self):
        global time_ind
        filestr = 'trade_'+(self.genFileName(time_ind))
        tradefile = self.base_dir+filestr

        print(filestr)
        f = open(tradefile, 'rb')
        return pickle.load(f).reset_index()
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
        for x in range(0 , tradedf.shape[0]):
            dic = {'pair': pair, 'price': tradedf.loc[x,'price'], 'qty' : int(tradedf.loc[x,'base_amount']) , 'time': int(tradedf.loc[x , 'time_exchange']*1000), 'isBuyerMaker': int(tradedf.loc[x , 'taker_side'])}
            blist.append(dic)
        return blist

    def createListFromLobDF(self, lobdf):
        # arrayTrades = np.zeros(len(listTrades), 4)
        blist = []
        lobdf = remove_duplicat_price_bidask(lobdf)
        bid_lob = lobdf.loc[lobdf['type']==1].reset_index(drop=True).sort_values(by='price', ascending = False)
        ask_lob = lobdf.loc[lobdf['type']==0].reset_index(drop=True).sort_values(by='price', ascending = True)

        lob = bid_lob
        for x in range(0, lob.shape[0]):
            lst = [lob.loc[x,'price'], lob.loc[x,'amount'], int(lob.loc[x,'type'])]
            blist.append(lst)
        alist = []
        lob = ask_lob
        for x in range(0, lob.shape[0]):
            lst = [lob.loc[x,'price'], lob.loc[x,'amount'], int(lob.loc[x,'type'])]
            alist.append(lst)
        dic = {'bids':blist, 'asks':alist, 'current_time': lob.loc[x,'date']}
        return dic

    def do_GET(self):
        global g_trade
        global time_ind
        # Process an HTTP GET request and return a response with an HTTP 200 status.
        self.send_response(requests.codes.ok, "thanks for contacting us")
        self.end_headers()

        print('time ind'+str(time_ind))
        #response = io.BytesIO()
        #response.write(b'This is POST request. ')
        #response.write(b'Received: ')
        if (g_trade):
            trades = self.get_trades()
            ltrades = self.createListFromTradesDF(trades)

        else:
            lob = self.get_lob()
            ltrades = self.createListFromLobDF(lob)
        s = json.dumps(ltrades)
        if len(ltrades) != 0:
            if g_trade:
                print(ltrades[-1])
            else:
                print(ltrades['bids'][-1])
        self.wfile.write(s.encode('utf-8'))
        time_ind += 1
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
    def setup_class(cls, trade=True):
        # Configure mock server.
        if trade:
            cls.mock_server_port = 65269
        else:
            cls.mock_server_port = 65110
        print('using port: ' + str(cls.mock_server_port))
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
    global g_trade
    g_trade = False
    if len(sys.argv)>1:
        pair = sys.argv[1]
        if len(sys.argv)>2:
            g_trade = sys.argv[1] != 'lob'
            print('Running as Trade Server')
        # ws = getBitmexWs(sys.argv[1])
    # else:
    #     ws = getBitmexWs(pair)
    t = TestMockServer(pair)
    t.setup_class(g_trade) # if false its a LOB mock service
    time.sleep(2)
    t.test_request_response()
    while True:
        time.sleep(1)

time_ind = 0
g_trade = None
pair = ''
