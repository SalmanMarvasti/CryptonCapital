import numpy as np
import io
# pd.to_datetime(df['date'],unit='s')
# https://401165704174.signin.aws.amazon.com/console
# salman
# %iCgxxG!R[Qe

import boto.s3.connection
from boto.exception import S3ResponseError
import sys, os
import pandas as pd
access_key = 'AKIAJW3Q6QOU2DLIWVVA'
secret_key = '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'
from boto.s3.connection import S3Connection
DOWNLOAD_LOCATION_PATH =  "d:/s3-backup/"
#DOWNLOAD_LOCATION_PATH = os.path.expanduser("~") +"/media/oem/CF7C-A41D" + "/s3-backup/"

import tarfile
import gzip
import matplotlib.pyplot as plt
import csv
directory = os.fsencode(DOWNLOAD_LOCATION_PATH)
#tempdf = pd.read_csv(DOWNLOAD_LOCATION_PATH+'binance.csv')
#validpairs = tempdf['PairNameApi']


def bid_ask_spread(df):
    bids = df.loc[df['type'] == 'b']
    asks = df.loc[df['type'] == 'a']
    return bids.nlargest(1, 'price'), asks.nlargest(1,'price')

def reformatdf(df):
    return df.drop(columns=['type'])

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
        strpath=strpath.replace('\\', '/')
        s=strpath.split('/')[-2:]
        date_prefix = '2018-'+s[0]+'-'+s[1]+' '

def convert_utc_to_epoch(timestamp_string):
    '''Use this function to convert utc to epoch'''
    temp = timestamp_string.split('.')
    # timestamp_string = temp[0]
    con = date_prefix+temp[0]
    timestamp = datetime.strptime(con, '%Y-%m-%d %H:%M:%S')

    epoch = int(calendar.timegm(timestamp.utctimetuple()))
    epoch = epoch + float('.'+temp[1])
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
    df = f.iloc[:,[0, -3,-2,-1]]  # ['date','price','base_amount','taker_size']
    df.loc[:,'time_exchange'] = df['time_exchange'].apply(convert_utc_to_epoch_trades)
    col = df.loc[:,'taker_side'] == 'SELL'
    df.loc[:,'taker_side'] = col.astype(int)


    with open(outFile, "w") as file:
        w = csv.writer(file)
        w.writerow(['date','price', 'amount', 'sell'])
        w.writerows(df.values)






def rewrite_cointick(inFile, outFile):
    df = pd.read_csv(inFile, delimiter=';')
    ob = df.loc[df['update_type'] == 'SNAPSHOT']

    ot = df.loc[df['update_type'] != 'SNAPSHOT']
    new_ot = pd.DataFrame(
        {'date': ot.time_exchange.apply(convert_utc_to_epoch), 'type': ot.is_buy, 'price': ot.entry_px,
         'amount': ot.entry_sx}, columns=['date', 'type', 'price', 'amount'])

    new_ob = pd.DataFrame(
        {'date': ob.time_exchange.apply(convert_utc_to_epoch), 'type': ob.is_buy, 'price': ob.entry_px,
         'amount': ob.entry_sx}, columns=['date', 'type', 'price', 'amount'])
    #  new_ob.type = [0 if x else 1 for x in new_ob.type]
    nob = new_ob.set_index('price')
    cur_dates = new_ot['date'].unique()
    x = 0
    print('going through dates')
    output_path = DOWNLOAD_LOCATION_PATH + 'ob/' + outFile
    with open(output_path, "w") as file:
        w = csv.writer(file)
        w.writerow(['price', 'date', 'type', 'amount'])
        update_addsub = ot.update_type

        for cdate in cur_dates:
            temp_ot = new_ot.loc[new_ot.date == cdate]
            temp_ot = temp_ot.set_index('price')
            temp_ot_up = update_addsub.loc[new_ot.date == cdate]

            for i in range(0, len(temp_ot)):
                row_to_append = temp_ot.iloc[i]
                p = temp_ot.index[i]  # this rows price

                if nob.index.contains(p):  # assume buy and ask correctly match
                    urow_to_append = temp_ot_up.iloc[i]
                    row_to_append['date'] = 0
                    row_to_append['type'] = 0
                    if urow_to_append == 'ADD':
                        nob.loc[p] += row_to_append  # ignores index column
                    else:
                        if urow_to_append == 'SUB':
                            nob.loc[p] -= row_to_append
                else:
                    nob.loc[p] = row_to_append
            nob['date'] = cdate
            x = x + 1
            # print('writing nob' + str(x))
            # csv_writer.writerow(nob.reset_index().values.tolist())
            tnob = nob.reset_index()
            w.writerows(tnob.values)

            # if (x<nArray.shape[0]):
            #     nArray[x:x+tnob.shape[0],0:] = tnob.values
            # else:
            #     print('warning slow append')
            #     nArray = np.append(nArray, tnob.values)
            # x = x + nob.shape[0]

    print('finished appending')


if __name__ == "__main__":

    #outFile = 'NEW_BITMEX_PERP_BTC_USD2.csv'
    #inFile = 'BITMEX_PERP_BTC_USD.csv'
    runTrades = False
    runOrderBook = True

    tradepairfile = 'PERP_BTC'

    if runTrades:
        for root, dirs, files in os.walk('/media/oem/79EF-A9BE/bitmex/trades/'):
            for filename in files:
                filename = os.fsdecode(filename)
                gen_date_prefix(root)
                if filename.find(tradepairfile) > -1:
                    rewrite_cointick_trades(root+'/'+filename, '/media/oem/79EF-A9BE/bitmex/converted_trades/'+date_prefix+filename[:-3])

    if runOrderBook:

        for root, dirs, files in os.walk('e:/bitmex/orderbook/'):

            for filename in files:
                filename = os.fsdecode(filename)
                gen_date_prefix(root) # generate the month day for datetime from folder
                if filename.find(tradepairfile)>-1:
                    rewrite_cointick(root+'/'+filename, date_prefix+filename[:-3])


    # df = pd.DataFrame(data = nArray[0:x,:], columns=['date', 'type', 'price', 'amount'])
    #fd.to_csv(DOWNLOAD_LOCATION_PATH+'ob/'+outFile, columns=['date', 'type', 'price', 'amount'])
    #print(ob.is_buy)
    #print(new_ob)
    # rec_check_dir(DOWNLOAD_LOCATION_PATH, gen_order_book_tar('EOSSDT','10', '2018'))