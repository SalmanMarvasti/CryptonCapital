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
DOWNLOAD_LOCATION_PATH = os.path.expanduser("~") + "/s3-backup/"
import tarfile
import gzip
import matplotlib.pyplot as plt
directory = os.fsencode(DOWNLOAD_LOCATION_PATH)
tempdf = pd.read_csv(DOWNLOAD_LOCATION_PATH+'binance.csv')
validpairs = tempdf['PairNameApi']


def inst_vwap(x):
    return sum(x.price * x.amount)/sum(x.amount)
# assume exchange is binance
def gen_order_book_file(pair, day, month, year, trade):
    ex = 'Binance'
    bnfxt = '_ob_10_'
    return ex + '_' + pair+bnfxt+year+'_'+month + '_' + day

def gen_order_book_tar(pair, month, year):
    ex = 'Binance'
    bnfxt = '_ob_10_'
    return ex + '_' + pair+bnfxt+year+'_'+month+'.tar'


def gen_pdf(xhist):
    hist_dist = scipy.stats.rv_histogram(xhist)


def rec_check_dir(x, fil):
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
                for cur_date in cur_dates:
                    dataFrame = df[dstr]
                    cf = dataFrame.loc[dataFrame['date'] == cur_date]
                    cr = cf['type'].replace('a', 'red')
                    cr = cr.replace('b', 'blue').tolist()
                    plt.clf()
                    histprice = cf['price']
                    if filesplit[1].endswith('BTC'):
                        histprice = histprice.multiply(10000)
                    print('size %s %s', histprice.size, len(cr))
                    N, bins, patches = plt.hist(histprice, histprice.size, weights=cf['amount'])
                    plt.title(str(text_dates[dcount])+' vp'+str(inst_vwap(cf)))
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


rec_check_dir(DOWNLOAD_LOCATION_PATH, gen_order_book_tar('ADAUSDT','07', '2018'))