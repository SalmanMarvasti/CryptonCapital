
from bitmex_websocket_cust import BitMEXWebsocket

import logging

from time import sleep





# Basic use of websocket.

def getBitmexWs(symb='XBTUSD'):




    # Instantiating the WS will make it connect. Be sure to add your api_key/api_secret.

    ws = BitMEXWebsocket(endpoint="wss://www.bitmex.com/", symbol=symb,

                         api_key='aHujwNKFyI3sxd5mKffKSf5O', api_secret='L3cVchQUvGnzJCxc_X8jIxuCj0dEs5IDyMmxYRtbZNkyMOrA')



    return ws





def setup_logger():

    # Prints logger info to terminalS
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)  # Change this to DEBUG if you want a lot more info


    ch = logging.StreamHandler()

    # create formatter

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # add formatter to ch

    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger





if __name__ == "__main__":
    ws = getBitmexWs()
    logger = setup_logger()
    logger.info("Instrument data: %s" % ws.get_instrument())

    while(ws.ws.sock.connected):

        logger.info("Ticker: %s" % ws.get_ticker())

        #if ws.api_key:

            #logger.info("Funds: %s" % ws.funds())

        logger.info("Market Depth: %s" % ws.market_depth())

        logger.info("Recent Trades: %s\n\n" % ws.recent_trades())

        sleep(2)
