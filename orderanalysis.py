import numpy as np
import io
# pd.to_datetime(df['date'],unit='s')
# https://401165704174.signin.aws.amazon.com/console
# salman
# %iCgxxG!R[Qe

#import boto.s3.connection
#from boto.exception import S3ResponseError
import sys, os
import pandas as pd
access_key = 'AKIAJW3Q6QOU2DLIWVVA'
secret_key = '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'
#from boto.s3.connection import S3Connection
DOWNLOAD_LOCATION_PATH = os.path.expanduser("~") + "/s3-backup/"
from modellingmanager import modelob

#DOWNLOAD_LOCATION_PATH = "/media/oem/CF7C-A41D" + "/s3-backup/"

import tarfile
import gzip
import matplotlib.pyplot as plt
import csv
directory = os.fsencode(DOWNLOAD_LOCATION_PATH)
#tempdf = pd.read_csv(DOWNLOAD_LOCATION_PATH+'binance.csv')
validpairs = ['XBTUSD', 'BTCUSDT', 'ETHUSD', 'ETHUSDT']


def bid_ask_spread(df):
    bids = df.loc[df['type'] == 'b']
    asks = df.loc[df['type'] == 'a']
    return bids.nlargest(1, 'price'), asks.nlargest(1,'price')

def reformatdf(df):
    return df.drop(columns=['type'])

def convert_df_bins(d2, bins):
    # d2['bins'] =
    return d2.groupby(pd.cut(d2['price'], bins)).sum()

def calc_diff(cf, pf, bins):
    bids = convert_df_bins(reformatdf(cf.loc[cf['type'] == 'b']), bins)
    asks = convert_df_bins(reformatdf(cf.loc[cf['type'] == 'a']), bins)
    pbids = convert_df_bins(reformatdf(pf.loc[pf['type'] == 'b']), bins)
    pasks = convert_df_bins(reformatdf(pf.loc[pf['type'] == 'a']), bins)
    a = asks.sub(pasks)
    b = bids.sub(pbids)
    return b,a



def calculate_bins_ticksize(p):
    x = np.histogram_bin_edges(p, 'fd')
    return x





def inst_vwap(x):
    return sum(x.price * x.amount)/sum(x.amount)
# assume exchange is binance
def gen_order_book_file(pair, day, month, year, trade):
    ex = 'Binance'
    bnfxt = '_ob_10_'
    return ex + '_' + pair+bnfxt+year+'_'+month + '_' + day # +CSV.GZ

def gen_order_book_tar(pair, month, year):
    ex = 'Binance'
    bnfxt = '_ob_10_'
    return ex + '_' + pair+bnfxt+year+'_'+month+'.tar'


# def gen_pdf(xhist):
#    hist_dist = scipy.stats.rv_histogram(xhist)


def rec_check_dir(x, fil):
    diff_mode = True
    for root, dirs, files in os.walk(x):
        for filename in files:
            filename = os.fsdecode(filename)
            filesplit = str(filename).split('_')
            if (filename.endswith(fil)) and validpairs.str.contains(filesplit[1]).any():
                df = untarfile(filename, root)
                dstr = list(df.keys())[0] # include a random date here
                cur_dates = df[dstr]['date'].unique()
                cur_dates.sort()
                text_dates = pd.to_datetime(cur_dates, unit='ms')

                dcount = 0
                pf = pd.DataFrame()
                for cur_date in cur_dates:
                    dataFrame = df[dstr]
                    cf = dataFrame.loc[dataFrame['date'] == cur_date]
                    if dcount>0 and diff_mode and dcount< len(cur_dates):
                        cprices = cf.price.as_matrix()
                        bins = calculate_bins_ticksize(cprices)
                        bids_diff, asks_diff = calc_diff(cf, bins) # calculate arrival between previous and current dataframes
                        amounts_diff = pd.concat([bids_diff.price, asks_diff.price])
                    else:
                        amounts_diff = cf['amount']

                    pf = cf
                    cr = cf['type'].replace('a', 'red')
                    cr = cr.replace('b', 'blue').tolist()
                    plt.clf()
                    histprice = cf['price']
                    if filesplit[1].endswith('BTC'):
                        histprice = histprice.multiply(10000)
                    print('size {0} {1}'.format(histprice.size, len(cr)))
                    N, bins, patches = plt.hist(histprice, histprice.size, weights=amounts_diff)
                    b, a = bid_ask_spread(cf)
                    print('bid ask'+str(b.price.tolist()[0])+' '+str(a.price.tolist()[0]))
                    plt.title(str(text_dates[dcount])+' vp'+str(inst_vwap(cf))+'ba'+str(b.price.tolist()[0]))
                    cc = 0
                    for p in patches:
                        if cc < len(cr):
                            p.set_facecolor(str(cr[cc]))
                        else:
                            p.set_facecolor('blue')
                        cc = cc + 1

                    savename = str(filesplit[1])+str(cur_date)+'.png'
                    print('saving'+savename)
                    plt.savefig(DOWNLOAD_LOCATION_PATH+'img/hist'+savename)
                    dcount = dcount + 1
                exit()
            else:
                continue

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


def untarfile(file, dir):
    os.chdir(dir)
    tar = tarfile.open(file)
    dic = {}
    for member in tar.getmembers():
        dstr = str(member).split('_')[-1]
        dstr = dstr.split('.')[0]
        f = tar.extractfile(member)
        f = gzip.GzipFile(fileobj = f)
        df = pd.read_csv(f)
        dic[dstr] = df
    return dic
    tar.close()


import calendar, time
from datetime import datetime


date_prefix = '2018-11-1 '
def gen_date_prefix(strpath):
    global date_prefix
    if(strpath):
        strpath=strpath.replace('\\', '/') # unify win and ubuntu
        s=strpath.split('/')[-2:]
        date_prefix = '2018-'+s[0]+'-'+s[1]+' '

import pytz
def convert_utc_to_epoch(timestamp_string):
    '''Use this function to convert utc to epoch'''
    temp = timestamp_string.split('.')
    timestamp_string = temp[0]
    # timestamp_string.rstrip('0')

    con = date_prefix+timestamp_string
    timestamp = datetime.strptime(con, '%Y-%m-%d %H:%M:%S')

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



def rewrite_cointick_trades(inFile, outFile):
    print('processing {}'.format(inFile))
    f = pd.read_csv(inFile, delimiter=';')
    df = f.iloc[:,[0, -3,-2,-1]]  # ['date','price','base_amount','taker_side']
    df.loc[:,'time_exchange'] = df['time_exchange'].apply(convert_utc_to_epoch_trades)
    col = df.loc[:,'taker_side'] == 'SELL'
    df.loc[:,'taker_side'] = col.astype(int)
    with open(outFile, "w") as file:
        w = csv.writer(file)
        w.writerow(['date','price', 'amount', 'sell'])
        w.writerows(df.values)


def just_convert_dates(inFile, outFile):
    first = True
    output_path = DOWNLOAD_LOCATION_PATH + 'date_orderbook/' + outFile
    # gc.disable()
    with open(output_path, "w") as file:
        # s = time.time()
        countn = 0
        for df in pd.read_csv(inFile, delimiter=';', chunksize=800):
            df['time_exchange'] = df.time_exchange.apply(convert_utc_to_epoch)
            df.to_csv(file, header=first, index=False)
            first = False


def dateparse (time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))

def read_trades():
    x = pd.read_csv('data.csv',delimiter=',', parse_dates=True,date_parser=dateparse, index_col='date')
    return x

import time
import gc

def rewrite_cointick_chunk(inFile, outFile,save_plots=False):


    first = True
    output_path = DOWNLOAD_LOCATION_PATH + 'ob/' + outFile
    # gc.disable()
    with open(output_path, "w") as file:
        w = csv.writer(file)
        w.writerow(['price', 'type', 'date', 'amount'])
        s = time.time()
        countn = 0
        for df in pd.read_csv(inFile, delimiter=';', chunksize=400):
            if first:
                ob = df.loc[df['update_type'] == 'SNAPSHOT']
                new_ob = pd.DataFrame({'date': ob.time_exchange.apply(convert_utc_to_epoch), 'type': ob.is_buy, 'price': ob.entry_px, 'amount': ob.entry_sx}, columns=['date', 'type', 'price', 'amount'])
                first = False
                nob = new_ob.set_index(['price','type'])
                df = df.loc[df['update_type'] != 'SNAPSHOT']
                nob.sort_index(inplace=True)
            nob = rewrite_cointick(inFile,outFile,df,nob,w,save_plots)
            countn +=1
            if(countn%5==0):
                etime = time.time()
                print('timed 30 runs' + str(etime-s))
                s=etime

    print('finished appending')


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
                                    raise ValueError('error cant have negative')
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
                            nob = return_bids_asks_cointick(inFile, outFile, snapshot_removed_ot, nob, w, save_plots)
                            return nob
                        if row_to_append.amount !=0:
                            raise ValueError("cannot subtract from nonexistant price"+str(p)+urow_to_append+' date'+str(text_dates[x]))



        nob.loc[:,'date'] = cdate
        nob.sort_index(inplace=True)

        if (cdate - prevdate)>1:
            nob = nob.loc[nob.amount != 0]
            rnob = nob.reset_index()
            if count_hist>2000:
                save_plots = False # Temp remove TODO

            if save_plots:
                nobprice = rnob.price.values
                nobamount = rnob['amount'].values
                plt.clf()
                N, hbins, patches = plt.hist(nobprice, len(nobprice), weights=nobamount)
                plt.title(str(text_dates[x]) + ' vp')  # +str(inst_vwap(nob))s
                colorhistbars(patches, rnob['type'].values)
                plt.savefig(DOWNLOAD_LOCATION_PATH + 'img/hist' + outFile[:-3] + str(count_hist) + '.png')
                count_hist += 1
            w.writerows(rnob.values)
            print('writing nob' + str(x)+' '+str(text_dates[x]))
            prevdate = cdate
        x = x + 1
    return nob



count_hist = 0
prevdate=0
def rewrite_cointick(inFile, outFile, ot, nob, w,save_plots=False):
    # df = pd.read_csv(inFile, delimiter=';')
    global count_hist
    global prevdate

    # df = df.loc[df['update_type'] != 'SNAPSHOT']

    new_ot = pd.DataFrame(
        {'date': ot.time_exchange.apply(convert_utc_to_epoch), 'type': ot.is_buy, 'price': ot.entry_px,
         'amount': ot.entry_sx, 'update_type': ot.update_type}, columns=['date', 'type', 'price', 'amount', 'update_type'])
    new_ot.sort_values('date', kind='mergesort', inplace=True)


    #  new_ob.type = [0 if x else 1 for x in new_ob.type]

    # noblookup = dict(zip(new_ob.price.values, new_ob.index.values))
    cur_dates = new_ot['date'].unique()
    text_dates = pd.to_datetime(cur_dates, unit='s')
    x = 0
    print('going through dates')

    if True:

        # update_addsub = ot.update_type


        for cdate in cur_dates:
            temp_ot = new_ot.loc[new_ot.date == cdate]
            #temp_ot = temp_ot.set_index('price')
            #temp_ot_up = update_addsub.loc[new_ot.date == cdate]
            #if(temp_ot_up.shape[0]!=temp_ot.shape[0]):
            #    raise ValueError("mismatched shape for update")




            # temp_ot_add = temp_ot.loc[temp_ot['update_type']=='ADD'].groupby(['price']).sum()
            # temp_ot_sub = temp_ot.loc[temp_ot['update_type']=='SUB'].groupby(['price']).sum()
            # m=pd.merge(nob, temp_ot_add, left_index=True,right_index=True, how='outer')
            # m.fillna(0,inplace=True)
            # m.loc[:, 'amount'] = m.loc[:, ['amount_x', 'amount_y']].sum(axis=1)
            # #
            # #
            # #
            # nob = m[['date_x','type_x', 'amount']]
            # nob.rename(columns={'date_x':'date','type_x':'type'}, inplace=True)
            # # nob['date'] = cdate
            # m = pd.merge(nob, temp_ot_sub, left_index=True, right_index=True, how='outer')
            # m.fillna(0, inplace=True)
            # m.loc[:, 'amount'] = m.loc[:, 'amount_x'] - m.loc[:, 'amount_y']
            # nob = m[['date_x','type_x', 'amount']]
            #
            # nob.rename(columns={'date_x':'date','type_x':'type'}, inplace=True)


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
                                    raise ValueError('error cant have negative')
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
                            nob = rewrite_cointick(inFile, outFile, snapshot_removed_ot, nob, w, save_plots)
                            return nob
                        if row_to_append.amount !=0:
                            raise ValueError("cannot subtract from nonexistant price"+str(p)+urow_to_append+' date'+str(text_dates[x]))



        nob.loc[:,'date'] = cdate
        nob.sort_index(inplace=True)

        # csv_writer.writerow(nob.reset_index().values.tolist())
        #if cdate%30==0:
        #    nob = nob.loc[nob.amount!=0]
        #nob.to_csv(file, header=False, chunksize=10000)



        if (cdate - prevdate)>1:
            nob = nob.loc[nob.amount != 0]
            rnob = nob.reset_index()
            if count_hist>2000:
                save_plots = False # Temp remove TODO

            if save_plots:
                nobprice = rnob.price.values
                nobamount = rnob['amount'].values
                plt.clf()
                N, hbins, patches = plt.hist(nobprice, len(nobprice), weights=nobamount)
                plt.title(str(text_dates[x]) + ' vp')  # +str(inst_vwap(nob))s
                colorhistbars(patches, rnob['type'].values)
                plt.savefig(DOWNLOAD_LOCATION_PATH + 'img/hist' + outFile[:-3] + str(count_hist) + '.png')
                count_hist += 1
            w.writerows(rnob.values)
            print('writing nob' + str(x)+' '+str(text_dates[x]))
            prevdate = cdate
        x = x + 1
    return nob
    # if (x<nArray.shape[0]):
    #     nArray[x:x+tnob.shape[0],0:] = tnob.values
    # else:
    #     print('warning slow append')
    #     nArray = np.append(nArray, tnob.values)
    # x = x + nob.shape[0]



if __name__ == "__main__":

    #outFile = 'NEW_BITMEX_PERP_BTC_USD2.csv'
    #inFile = 'BITMEX_PERP_BTC_USD.csv'
    runTrades = False
    runOrderBook = True # d:/bitmex/orderbook/11/03

    tradepairfile = 'PERP_BTC'

    if runTrades:
        for root, dirs, files in os.walk('/media/oem/79EF-A9BE/bitmex/trades/'):
            for filename in files:
                filename = os.fsdecode(filename)
                gen_date_prefix(root)
                if filename.find(tradepairfile) > -1:
                    rewrite_cointick_trades(root+'/'+filename, '/media/oem/79EF-A9BE/bitmex/converted_trades/'+date_prefix+filename[:-3])

    if runOrderBook:
        # media/oem/79EF-A9BE/
        for root, dirs, files in os.walk('/media/oem/79EF-A9BE/bitmex/orderbook/'):

            for filename in files:
                filename = os.fsdecode(filename)
                gen_date_prefix(root) # generate the month day for datetime from folder
                if filename.find(tradepairfile)>-1:
                    #rewrite_cointick_chunk(root+'/'+filename, date_prefix+filename[:-3],True)
                    just_convert_dates(root+'/'+filename, 'DATE_CHANGE_'+date_prefix+filename[:-3])


    # df = pd.DataFrame(data = nArray[0:x,:], columns=['date', 'type', 'price', 'amount'])
    #fd.to_csv(DOWNLOAD_LOCATION_PATH+'ob/'+outFile, columns=['date', 'type', 'price', 'amount'])
    #print(ob.is_buy)
    #print(new_ob)
    # rec_check_dir(DOWNLOAD_LOCATION_PATH, gen_order_book_tar('EOSSDT','10', '2018'))