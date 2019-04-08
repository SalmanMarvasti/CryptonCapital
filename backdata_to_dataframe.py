import pandas as pd
import numpy as np
import os
import pickle
import datetime
import pytz
import gzip
import matplotlib.pyplot as plt

def convert_utc_to_epoch(timestamp_string):
    '''Use this function to convert utc to epoch'''
    temp = timestamp_string.split('.')
    timestamp_string = temp[0]
    # timestamp_string.rstrip('0')

    con = date_prefix+timestamp_string
    timestamp = datetime.datetime.strptime(con, '%Y-%m-%d %H:%M:%S')

    # timestamp = timestamp.replace(tzinfo=pytz.utc)
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

def colorhistbars(patches,cr):
    cc = 0
    for p in patches:
        if cc < len(cr):
            if cr[cc]==1:
                p.set_facecolor('red')
            else:
                p.set_facecolor('blue')
        else:
            p.set_facecolor('blue')
        cc = cc + 1


count_hist = 0
prevdate=0


def return_bids_asks_cointick(inFile, outFile, trades, ot, nob, w,save_plots=False):
    # df = pd.read_csv(inFile, delimiter=';')
    global count_hist
    global prevdate
    output_path = './ob/' + outFile
    output_trade = './ob/'+ 'trade_'+outFile
    # df = df.loc[df['update_type'] != 'SNAPSHOT']

    new_ot = pd.DataFrame(
        {'date': ot.time_exchange.apply(convert_utc_to_epoch), 'type': ot.is_buy, 'price': ot.entry_px,
         'amount': ot.entry_sx, 'update_type': ot.update_type}, columns=['date', 'type', 'price', 'amount', 'update_type'])
    new_ot.sort_values(['date','type'], kind='mergesort', inplace=True)


    #  new_ob.type = [0 if x else 1 for x in new_ob.type]

    # noblookup = dict(zip(new_ob.price.values, new_ob.index.values))
    cur_dates = new_ot['date'].unique()
    min_date = cur_dates.min()
    #condition = trades['time_exchange']>min_date and trades['time_exchange']>max_date
    #trades.loc[condition]
    cur_trades = []
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

                if nob.index.isin([p]).any() and urow_to_append[0:3]!='SNA':  # assume buy and ask correctly match
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
                                if nob.loc[p, 'amount']==0:
                                    print('Warning negative values here'+str(row_to_append.amount))
                                nob.loc[p,'amount'] -= (row_to_append[3])


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
                            nob, cur_trades = return_bids_asks_cointick(inFile, outFile, trades,snapshot_removed_ot, nob, w, save_plots)
                            return nob, cur_trades
                        else:
                            nob.loc[p, :] = row_to_append[[0, 3]]
                            if row_to_append.amount != 0 and row_to_append.type == 'SUB':
                                print('Warning could have negative values here')
                                nob.loc[p, ['amount']] *=-1
                if (cdate - prevdate)>22:
                    nob.loc[:, 'date'] = cdate
                    nob.sort_index(inplace=True)
                    print('time diff'+str((cdate - prevdate)))
                    cur_trades = trades[min_date - 2:cdate + 1]
                    nob = remove_duplicat_price_bidask(nob)
                    nob['type'] = (nob['amount']<0).apply(int)
                    nob['amount'] = np.abs(nob.amount)
                    nob = nob.loc[nob.amount > 0.1]
                    nob.set_index(['price', 'type'], inplace=True)
                    rnob = nob.reset_index()
                    if count_hist>2000:
                        save_plots = False # Temp remove TODO

                    if save_plots:
                        nobprice = rnob.price.values
                        nobamount = rnob['amount'].values
                        plt.clf()
                        mn = min(nobprice)
                        mx = max(nobprice)
                        N, hbins, patches = plt.hist(nobprice,bins=np.linspace(mn, mx, int((mx-mn)/0.5)+1), weights=nobamount)
                        plt.title(str(text_dates[x]) + ' vp')  # +str(inst_vwap(nob))s
                        colorhistbars(patches, rnob['type'].values)
                        plt.savefig('./img/hist' + outFile[:-3] + str(count_hist) + '.png')

                        nob.to_pickle(output_path[:-4] + str(count_hist))
                        cur_trades.to_pickle(output_trade[:-4] + str(count_hist))
                    count_hist += 1
                    prevdate = cdate
        x = x + 1

    return nob,cur_trades


def remove_duplicat_price_bidask(nob):
    aaa = nob.reset_index()
    aaa.loc[aaa.type == 1, ['amount']] = aaa[aaa.type == 1].amount * -1
    bbb = aaa.groupby(['price','date']).sum()
    filter1 = bbb.amount<0
    filter0 = bbb.amount>0
    min_len = min(sum(filter0), sum(filter1))
    if min_len>15:
        min_len = 15
    npind1 = filter1.nonzero()[0][0:max(min_len,sum(filter1)-3)  + 1]
    npind0 = filter0.nonzero()[0][0:min_len + 1]
    npind = np.append(npind1, npind0)
    bbb = bbb.iloc[npind, :]
    return bbb.reset_index()

# def remove_duplicat_price_bidask(nob):
#     aaa = nob.reset_index()
#     filter1 = aaa.type == 1
#     aaa.loc[filter1, ['amount']] = aaa[filter1].amount * -1 # make negative so summation would work
#     bbb = aaa.groupby(['price', 'date']).sum()
#     bbb = bbb.reset_index()
#     bbb.loc[bbb.type == 1, ['amount']] = bbb[bbb.type == 1].amount * -1
#     filt = bbb.amount<0
#     if filt.any():
#         ccc=bbb.loc[filt, ['type', 'amount']]
#         ccc.amount*=-1
#         ccc['type'] = (ccc.type+1)%2
#         bbb.loc[filt, ['type', 'amount']] = ccc
#     filter1 = bbb.type == 1
#     filter0 = bbb.type == 0
#     min_len = min(sum(filter0), sum(filter1))
#     if min_len>20:
#         min_len = 20
#     npind1 = filter1.nonzero()[0][0:min_len + 1]
#     npind0 = filter0.nonzero()[0][0:min_len + 1]
#     npind = np.append(npind1, npind0)
#     bbb = bbb.iloc[npind, :]
#     return bbb


import time
def rewrite_cointick_chunk(basepath, lobfile, outFile, save_plots=False):
    first = True
    output_path = './ob/' + outFile
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
        for df in pd.read_csv(inFile, delimiter=';', chunksize=1000):
            if first:
                ob = df.loc[df['update_type'] == 'SNAPSHOT']
                new_ob = pd.DataFrame({'date': ob.time_exchange.apply(convert_utc_to_epoch), 'type': ob.is_buy, 'price': ob.entry_px, 'amount': ob.entry_sx}, columns=['date', 'type', 'price', 'amount'])
                first = False
                nob = new_ob.set_index(['price','type'])
                df = df.loc[df['update_type'] != 'SNAPSHOT']
                nob.sort_index(inplace=True)
            nob,curtrade = return_bids_asks_cointick(inFile,outFile,trade_df, df,nob,file,save_plots)

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