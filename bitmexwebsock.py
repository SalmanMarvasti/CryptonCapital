
from bitmex_websocket_cust import BitMEXWebsocket

import logging

from time import sleep





# Basic use of websocket.

def getBitmexWs(symb='XBTUSD'):
    #                      api_key='aHujwNKFyI3sxd5mKffKSf5O', api_secret='L3cVchQUvGnzJCxc_X8jIxuCj0dEs5IDyMmxYRtbZNkyMOrA')

    # Instantiating the WS will make it connect. Be sure to add your api_key/api_secret.

    ws = BitMEXWebsocket(endpoint="wss://www.bitmex.com/", symbol=symb,
                         api_key='hlZurVuOwx1NqIUlYorfKLn3', api_secret='FMgTYMmc8wKU6B4MrPJ9FGnDSWisJdn9J_rCeCJhpOTafX2s')

    # Instantiating the WS will make it connect. Be sure to add your api_key/api_secret.
    # API URL.
    # BASE_URL = "wss://testnet.bitmex.com/api/v1/"
    # # BASE_URL = "https://www.bitmex.com/api/v1/" # Once you're ready, uncomment this.
    #
    # # The BitMEX API requires permanent API keys. Go to https://testnet.bitmex.com/app/apiKeys to fill these out.
    # API_KEY = "LvFUQxoAisotHBG9_TJP7Vnt"
    # API_SECRET = "nbt2JwSNAnielJCq4x2FJC3dTNLxCTZtHeGCeYoro2lWbjak"

    # ws = BitMEXWebsocket(endpoint=BASE_URL, symbol=symb,
    #
    #                      api_key=API_KEY,
    #                      api_secret=API_SECRET)


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
