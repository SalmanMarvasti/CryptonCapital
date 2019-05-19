import sys
from time import sleep
from time import time
import pandas as pd
from market_maker.market_maker import OrderManager
import pickle
import uuid
import base64
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='./custom_strategy.log',
                    filemode='w')

def prepareAndSetID(order, order2, order3=[]):
    str = base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n')
    order[-1]['clOrdID'] = "mm_bitmex_" + str
    str = base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n')
    order2[-1]['clOrdID'] = "mm_bitmex_" + str
    #str = base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n')
    if len(order3) > 1:
        order3[-2]['clOrdID'] = "mm_bitmex_" + str
    return order

class CustomOrderManager(OrderManager):
    """A sample order manager for implementing your own custom strategy"""
    tradingpair = 'XBTUSD'
    filepath = './'
    d = {}
    to_submit_sell_orders = {}
    to_submit_buy_orders = {}
    last_time = time()

    def place_orders(self) -> None:
        # implement your custom strategy here

        buy_orders = []
        sell_orders = []


        # populate buy and sell orders, e.g.

        file = open(self.filepath + 'df/' + self.tradingpair + 'trades.pkl', 'rb')
        dflist = pickle.load(file)
        file.close()
        ticker = self.exchange.get_ticker()
        mypos = self.exchange.get_position()
        print('MY POS:'+str(mypos))
        open_qty = mypos['currentQty']
        open_buys = mypos['openOrderBuyQty']
        open_sell = mypos['openOrderSellQty']
        print('OPEN POSITION QTY'+str(open_qty)+' limit buys:'+str(open_buys)+ 'limit sells:'+str(open_sell))
        bprice = float(ticker['buy'])
        aprice = float(ticker['sell'])
        actual_mid = (bprice+aprice) / 2

        # tmp_dict = (111, 5310.963348252087, -2.286651747913311, 0.6608195356969412, 5335.036651747913, 5292.875, 5330.25, 58.89052108574486)
        # dflist.append(tmp_dict)
        count = 0

        avg_price = 0
        avg_diff = 0
        avg_stop = 0
        mid = 0
        predicted_price = 0
        stoploss = 0
        qty = 30
        ll = len(dflist)
        if ll>0:
            logging.info('Trades suggested' + str(ll))

        for i in range(ll-1, -1, -1):
            tmp_dict = dflist[i]
            count = count + 1

            (validtill_timestamp, predicted_price, price_diff, confidence, stoploss, sma, mid, rsi) = (tmp_dict[0], tmp_dict[1], tmp_dict[2], tmp_dict[3], tmp_dict[4], tmp_dict[5], tmp_dict[6], tmp_dict[7])
            print(validtill_timestamp)

            #mid = actual_mid


            if validtill_timestamp in self.d.keys():
                if time()-self.last_time>90:
                    continue
                logging.info('retrying order')
                # if validtill_timestamp not in self.to_submit_sell_orders.keys():
                #     if validtill_timestamp not in self.to_submit_buy_orders.keys():
                #         continue
                #     else:
                #         buy_orders.append( self.to_submit_buy_orders[validtill_timestamp])
                # else:
                #     sell_orders.append( self.to_submit_sell_orders[validtill_timestamp])
                if(price_diff<0):
                    if predicted_price>bprice: # if ticker lower cancel
                        logging.info('cancelling repeat short')
                        continue
                    else:
                        buy_orders.append({'execInst':'ParticipateDoNotInitiate','price': round(predicted_price) - 0.5, 'orderQty': qty, 'side': "Buy"})
                if price_diff>1:
                    # predicted_price = aprice+15  # test code
                    # mid = actual_mid
                    # open_qty = 10

                    if predicted_price<aprice:
                        logging.info('cancelling repeat long')
                        continue
                    else:
                        sell_orders.append({'execInst':'ParticipateDoNotInitiate','price': round(predicted_price) + 0.5, 'orderQty': qty, 'side': "Sell"})


            #     print('ignoring duplicate')
            #     continue
            self.d.update([(validtill_timestamp, price_diff)])

            avg_diff+=price_diff
            avg_price+=mid
            avg_stop += stoploss

        if (len(sell_orders)>0 and open_qty>0) or (len(buy_orders)>0 and open_qty<0):
            logging.info('Take profit limit orders only no new orders')
            self.converge_orders(buy_orders, sell_orders)
            sleep(2)
            return
        else:
            sell_orders.clear()
            buy_orders.clear()

        if count == 0:
            count = 1
        price_diff = avg_diff/count
        stoploss = avg_stop/count
        if avg_price!=0 and abs(mid - avg_price/count)<5:
            logging.info('avg mid ok setting avg mid')
            mid = avg_price/count

        if mid!=0 and abs(mid - actual_mid)>15:
            logging.info('high difference between mid and actual'+str(actual_mid)+str(mid))
        else:
            logging.info('ok diff' +str(price_diff))


        if price_diff>0:
            print("placing new ")
            if abs(mid - actual_mid)<15:
                logging.info('++++++++++going long stop:'+str(stoploss)+' target:'+str(predicted_price))
                buy_orders.append({'execInst':'ParticipateDoNotInitiate','price': round(mid)-1, 'orderQty': qty, 'side': "Buy"})
                #self.to_submit_sell_orders.update([(validtill_timestamp, {'execInst':'LastPrice, ParticipateDoNotInitiate','price':round(predicted_price) +0.5, 'orderQty': qty, 'side': "Sell"})])
                sell_orders.append({'execInst':'LastPrice, ParticipateDoNotInitiate','price':round(predicted_price) +2.5, 'orderQty': qty, 'side': "Sell"})
                #sell_orders.append({'execInst':'LastPrice, ParticipateDoNotInitiate, ReduceOnly','ordType':'LimitIfTouched','stopPx': round(predicted_price)-0.5, 'price': round(predicted_price) +0.5, 'orderQty': qty, 'side': "Sell"})
                sell_orders.append({ 'execInst':'LastPrice,ParticipateDoNotInitiate, ReduceOnly', 'ordType':'StopLimit', 'stopPx':round(stoploss+10),'orderQty': qty,  'price': round(stoploss) - 25,  'side': "Sell"})
                prepareAndSetID(buy_orders, sell_orders, buy_orders)

        if price_diff<0:
            print("placing new ")
            if abs(mid - actual_mid) < 15:
                logging.info('-------going short mid, short:'+str(mid)+' '+str(stoploss)+' target:'+str(predicted_price))
                sell_orders.append({'execInst':'ParticipateDoNotInitiate', 'price': round(mid)+1, 'orderQty':qty, 'side': "Sell"})
                # self.to_submit_buy_orders.update([(validtill_timestamp, {'execInst':'LastPrice,ParticipateDoNotInitiate, ReduceOnly', 'price': round(predicted_price), 'orderQty': qty, 'side': "Buy"})])
                buy_orders.append({'execInst':'LastPrice,ParticipateDoNotInitiate', 'price': round(predicted_price)-2, 'orderQty': qty, 'side': "Buy"})
                #buy_orders.append({'execInst':'LastPrice,ParticipateDoNotInitiate, ReduceOnly', 'ordType':'LimitIfTouched','stopPx': round(predicted_price)+1.5, 'price': round(predicted_price), 'orderQty': qty, 'side': "Buy"})
                buy_orders.append({'execInst':'LastPrice, ParticipateDoNotInitiate,ReduceOnly', 'ordType':'StopLimit', 'stopPx' : round(stoploss)-0.5,'orderQty': qty,  'price': round(stoploss) +25,'side': "Buy"})
                prepareAndSetID(buy_orders, sell_orders, buy_orders)

        self.last_time = time()
        while True:
            try:
                self.converge_orders(buy_orders, sell_orders)
            except:
                logging.info("HTTPError raised. Retrying in 2 seconds...")
                sleep(2)
                continue
            break
        #self.converge_orders(buy_orders, sell_orders)
        sleep(1)



def run() -> None:
    order_manager = CustomOrderManager()

    # Try/except just keeps ctrl-c from printing an ugly stacktrace
    try:
        order_manager.run_loop()
    except (KeyboardInterrupt, SystemExit):
        sys.exit()
