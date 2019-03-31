import pandas as pd
import numpy as np
import os
import pickle
import datetime
import pytz
def convert_utc_to_epoch(timestamp_string):
    '''Use this function to convert utc to epoch'''
    temp = timestamp_string.split('.')
    timestamp_string = temp[0]
    # timestamp_string.rstrip('0')

    con = date_prefix+timestamp_string
    timestamp = datetime.datetime.strptime(con, '%Y-%m-%d %H:%M:%S')

    timestamp = timestamp.replace(tzinfo=pytz.utc)
    epoch = timestamp.timestamp()
    epoch = epoch + float('.' + temp[1])
    return epoch

import ciso8601
def convert_utc_to_epoch_trades(timestamp_string):
    '''Use this function to convert utc to epoch'''
    #temp = timestamp_string.split('.')
    #con = temp[0]
    #timestamp = datetime.strptime(con, '%Y-%m-%dT%H:%M:%S')
    timestamp = ciso8601.parse_datetime(timestamp_string)
    epoch = timestamp.timestamp()
    return epoch
count_hist = 0
prevdate=0

count_hist = 0
prevdate=0
def return_bids_asks_cointick(inFile, outFile, trades, ot, nob, w,save_plots=False):
    # df = pd.read_csv(inFile, delimiter=';')
    global count_hist
    global prevdate

    # df = df.loc[df['update_type'] != 'SNAPSHOT']

    new_ot = pd.DataFrame(
        {'date': ot.time_exchange.apply(convert_utc_to_epoch), 'type': ot.is_buy, 'price': ot.entry_px,
         'amount': ot.entry_sx, 'update_type': ot.update_type}, columns=['date', 'type', 'price', 'amount', 'update_type'])
    new_ot.sort_values(['date','type'], kind='mergesort', inplace=True)


    #  new_ob.type = [0 if x else 1 for x in new_ob.type]

    # noblookup = dict(zip(new_ob.price.values, new_ob.index.values))
    cur_dates = new_ot['date'].unique()
    max_date = cur_dates.max()
    min_date = cur_dates.min()-5
    #condition = trades['time_exchange']>min_date and trades['time_exchange']>max_date
    #trades.loc[condition]
    cur_trades = trades[min_date-10:max_date]

    text_dates = pd.to_datetime(cur_dates, unit='s')
    x = 0
    print('going through dates')

    if True:

        # update_addsub = ot.update_type


        for cdate in cur_dates:
            temp_ot = new_ot.loc[new_ot.date == cdate]
            #bids = temp_ot.loc[temp_ot.type==1]
            # asks = temp_ot.loc[temp_ot.type==0]

            for i in range(0, len(temp_ot)):
                row_to_append = temp_ot.iloc[i]
                p = (row_to_append.price, row_to_append.type)  # this rows price
                urow_to_append = row_to_append.update_type

                if nob.index.isin([p]).any() and nob.loc[p,'amount']!=0 and urow_to_append[0:3]!='SNA':  # assume buy and ask correctly match
                    if p[1]!= row_to_append.type:
                        # if nob.loc[p,'amount']!=0:
                        print('not_ok'+urow_to_append)

                    if urow_to_append[0] == 'A':
                        if p[1] != row_to_append.type:
                            print('error')
                            #nob.loc[p, ['type','amount']] = row_to_append[[ 1, 3]]
                        else:
                            nob.loc[p,'amount'] += row_to_append[3] # ignores index column
                    else:
                        if urow_to_append == 'SUB':
                            if p[1] != row_to_append.type:
                                raise ValueError('error nothing to subtract setting')
                                # nob.loc[p, ['type','amount']] = row_to_append[[ 1, 3]]
                            else:
                                if nob.loc[p, 'amount']==0 or  nob.loc[p,'amount']<row_to_append.amount:
                                    print('Warning negative values here')
                                nob.loc[p,'amount'] -= row_to_append[3]


                        else:
                            print('*****Setting' + urow_to_append +row_to_append.amount)
                            nob.loc[p, :] = row_to_append[[ 0, 3]]


                else:
                    if urow_to_append[0]=='A' or urow_to_append=='SET':
                        nob.loc[p,:] = row_to_append[[0,3]]
                    else:
                        if(urow_to_append == 'SNAPSHOT'):
                            nob = new_ot.loc[new_ot.update_type=='SNAPSHOT'].drop(['update_type'],axis=1)
                            if(nob.date.unique().shape[0]!=1):
                                raise ValueError("Snapshot from multiple dates"+str(p)+str(text_dates[x]))
                            print('Found SNAPSHOT in middle of conversion')
                            if nob.shape[0]<20:
                                raise ValueError("Snapshot split"+str(p)+str(text_dates[x]))
                            start_row = x+nob.shape[0]
                            snap_found_idx = 0
                            snap_not_found_idx = 0
                            for j in range(len(ot)):
                                if ot.update_type.iloc[j] == 'SNAPSHOT':
                                    snap_found_idx = j
                                else:
                                    if snap_found_idx > 9:
                                        if ot.update_type.iloc[j] != 'SNAPSHOT':
                                            snap_not_found_idx = j
                                            break
                            if snap_not_found_idx:
                                start_row = snap_not_found_idx
                            nob.set_index(['price', 'type'], inplace=True)
                            nob = nob.loc[nob.amount != 0]
                            snapshot_removed_ot = ot.iloc[start_row:-1,:]
                            nob = return_bids_asks_cointick(inFile, outFile, trades,snapshot_removed_ot, nob, w, save_plots)
                            return nob
                        if row_to_append.amount !=0:
                            raise ValueError("cannot subtract from nonexistant price"+str(p)+urow_to_append+' date'+str(text_dates[x]))



        nob.loc[:,'date'] = cdate
        nob.sort_index(inplace=True)

        if (cdate - prevdate)>20:
            nob = nob.loc[nob.amount != 0]
            rnob = nob.reset_index()
            if count_hist>2000:
                save_plots = False # Temp remove TODO

            if save_plots:
                nobprice = rnob.price.values
                # nobamount = rnob['amount'].values
                # plt.clf()
                # N, hbins, patches = plt.hist(nobprice, len(nobprice), weights=nobamount)
                # plt.title(str(text_dates[x]) + ' vp')  # +str(inst_vwap(nob))s
                # colorhistbars(patches, rnob['type'].values)
                # plt.savefig(DOWNLOAD_LOCATION_PATH + 'img/hist' + outFile[:-3] + str(count_hist) + '.png')
                # count_hist += 1
            # w.writerows(rnob.values)
            # print('writing nob' + str(x)+' '+str(text_dates[x]))
            prevdate = cdate
        x = x + 1
    return nob,cur_trades




import time
def rewrite_cointick_chunk(basepath, lobfile, outFile,save_plots=False):
    first = True
    output_path = './ob/' + outFile
    output_trade = './ob/'+ 'trade_'+outFile
    inFile = os.path.join(basepath, lobfile)
    tradepath = os.path.join(basepath, 'trades')
    tradefile = os.path.join(tradepath, lobfile)

    try:
        trade_df = pd.read_csv(tradefile, delimiter=';')

    except Exception as e:
        print('error loading trade gz file.. retrying'+str(e))
        trade_df = pd.read_csv(tradefile[:-3], delimiter=';') #print('processing trades')
    trade_df = trade_df.iloc[:,[0, -3,-2,-1]]  # ['time_exchange','price','base_amount','taker_side']
    trade_df.loc[:,'time_exchange'] = trade_df['time_exchange'].apply(convert_utc_to_epoch_trades)
    col = trade_df.loc[:,'taker_side'] == 'SELL'
    trade_df.loc[:,'taker_side'] = col.astype(int)

    trade_df = trade_df.set_index('time_exchange')
    # gc.disable()
    with open(output_path, "wb") as file:

        s = time.time()
        countn = 0
        for df in pd.read_csv(inFile, delimiter=';', chunksize=600):
            if first:
                ob = df.loc[df['update_type'] == 'SNAPSHOT']
                new_ob = pd.DataFrame({'date': ob.time_exchange.apply(convert_utc_to_epoch), 'type': ob.is_buy, 'price': ob.entry_px, 'amount': ob.entry_sx}, columns=['date', 'type', 'price', 'amount'])
                first = False
                nob = new_ob.set_index(['price','type'])
                df = df.loc[df['update_type'] != 'SNAPSHOT']
                nob.sort_index(inplace=True)
            nob,curtrade = return_bids_asks_cointick(inFile,outFile,trade_df, df,nob,file,save_plots)
            nob.to_pickle(output_path[:-4]+str(countn))
            curtrade.to_pickle(output_trade[:-4]+str(countn))
            countn +=1
            if(countn%5==0):
                etime = time.time()
                print('timed 30 runs' + str(etime-s))
                s=etime

    print('finished appending')



def gen_date_prefix(strpath):
    global date_prefix
    if(strpath):
        strpath=strpath.replace('\\', '/') # unify win and ubuntu
        s=strpath.split('/')[-2:]
        date_prefix = '2018-'+s[0]+'-'+s[1]+' '


if __name__ == "__main__":

    runTrades = False
    runOrderBook = True # d:/bitmex/orderbook/11/03

    tradepairfile = 'PERP_BTC'

    # if runTrades:
    #     for root, dirs, files in os.walk('d:\\bitmex\\ob\\'):
    #         for filename in files:
    #             filename = os.fsdecode(filename)
    #             gen_date_prefix(root)
    #             if filename.find(tradepairfile) > -1:
    #                 rewrite_cointick(root+'/'+filename, 'c:\\temp\\df\\'+date_prefix+filename[:-3])
    if runOrderBook:
        # media/oem/79EF-A9BE/
        for root, dirs, files in os.walk('d:\\bitmex\\ob\\12\\03'):

            for filename in files:
                filename = os.fsdecode(filename)
                gen_date_prefix(root) # generate the month day for datetime from folder
                if filename.find(tradepairfile)>-1 and root.find('trades')==-1:
                    filename
                    rewrite_cointick_chunk(root,filename, date_prefix+filename[:-3],True)
                    #just_convert_dates(root+'/'+filename, 'DATE_CHANGE_'+date_prefix+filename[:-3])


        print('completed folder')