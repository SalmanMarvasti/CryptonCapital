import sys
import pandas as pd
from market_maker.market_maker import OrderManager
import pickle
import uuid
import base64

def prepareAndSetID(order, order2, order3=[]):
    str = base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n')
    order[-1]['clOrdID'] = "mm_bitmex_" + str
    str = base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n')
    order2[-1]['clOrdID'] = "mm_bitmex_" + str
    str = base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n')
    if len(order3)>1:
        order3[-2]['clOrdID'] = "mm_bitmex_" + str
    return order

class CustomOrderManager(OrderManager):
    """A sample order manager for implementing your own custom strategy"""
    tradingpair = 'XBTUSD'
    filepath = './'
    d = {}


    def place_orders(self) -> None:
        # implement your custom strategy here

        buy_orders = []
        sell_orders = []

        # populate buy and sell orders, e.g.
        print("placing new ")
        file = open(self.filepath + 'df/' + self.tradingpair + 'trades.pkl', 'rb')
        list = pickle.load(file)
        file.close()
        ticker = self.exchange.get_ticker()
        mypos = self.exchange.get_position()
        print('MY POS:'+str(mypos))
        bprice = float(ticker['buy'])
        aprice = float(ticker['sell'])
        actual_mid = (bprice+aprice) / 2

        # tmp_dict = (111, 5310.963348252087, -2.286651747913311, 0.6608195356969412, 5335.036651747913, 5292.875, 5330.25, 58.89052108574486)
        # list.append(tmp_dict)
        count = 0

        avg_price = 0
        avg_diff = 0
        mid = 0
        predicted_price = 0
        stoploss = 0
        qty = 1000
        for i in range(len(list)-1, -1, -1):
            tmp_dict = list[i]
            count = count + 1
            if count > 3:
                continue

            (validtill_timestamp, predicted_price, price_diff, confidence, stoploss, sma, mid, rsi) = (tmp_dict[0], tmp_dict[1], tmp_dict[2], tmp_dict[3], tmp_dict[4], tmp_dict[5], tmp_dict[6], tmp_dict[7])
            print(validtill_timestamp)



            if validtill_timestamp in self.d.keys():
                 continue
            #     if(price_diff<0):
            #         if predicted_price>bprice: # if ticker lower cancel
            #             print('cancelling simulated stop')
            #             continue
            #         else:
            #             buy_orders.append({'price': round(predicted_price) - 0.5, 'orderQty': qty, 'side': "Buy"})
            #     if price_diff>1:
            #         if predicted_price<aprice:
            #             continue
            #         else:
            #             sell_orders.append({'price': round(predicted_price) + 0.5, 'orderQty': qty, 'side': "Sell"})
            #
            #     print('ignoring duplicate')
            #     continue
            self.d.update([(validtill_timestamp, price_diff)])

            avg_diff+=price_diff
            avg_price+=mid
        if count == 0:
            count = 1
        price_diff = avg_price/count
        if abs(mid - avg_price/count)<5:
            print('avg mid ok setting avg mid')
            mid = avg_price/count



        if price_diff>2:
            if mid <= actual_mid:
                print('++++++++++going long')
                buy_orders.append({'execInst':'ParticipateDoNotInitiate','price': round(mid)-1, 'orderQty': qty, 'side': "Buy"})
                sell_orders.append({'execInst':'LastPrice, ParticipateDoNotInitiate, ReduceOnly','ordType':'LimitIfTouched','stopPx': round(predicted_price)-0.5, 'price': round(predicted_price) +0.5, 'orderQty': qty, 'side': "Sell"})
                sell_orders.append({ 'execInst':'LastPrice, ReduceOnly', 'ordType':'StopLimit', 'stopPx':round(stoploss),'orderQty': qty,  'price': round(stoploss) - 1,  'side': "Sell"})
                prepareAndSetID(buy_orders, sell_orders, sell_orders)

        if price_diff<-2:
            if mid >= actual_mid:
                print('------------going short')
                sell_orders.append({'execInst':'ParticipateDoNotInitiate', 'price': round(mid)+1, 'orderQty':qty, 'side': "Sell"})
                buy_orders.append({'execInst':'LastPrice,ParticipateDoNotInitiate, ReduceOnly', 'ordType':'LimitIfTouched','stopPx': round(predicted_price)+1.5, 'price': round(predicted_price), 'orderQty': qty, 'side': "Buy"})
                buy_orders.append({'execInst':'LastPrice, ParticipateDoNotInitiate,ReduceOnly', 'ordType':'StopLimit', 'stopPx' : round(stoploss)-0.5,'orderQty': qty,  'price': round(stoploss) + 1,'side': "Buy"})
                prepareAndSetID(buy_orders, sell_orders, buy_orders)


        self.converge_orders(buy_orders, sell_orders)



def run() -> None:
    order_manager = CustomOrderManager()

    # Try/except just keeps ctrl-c from printing an ugly stacktrace
    try:
        order_manager.run_loop()
    except (KeyboardInterrupt, SystemExit):
        sys.exit()
