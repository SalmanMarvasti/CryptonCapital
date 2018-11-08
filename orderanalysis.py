import numpy as np
import io
#
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

def rec_check_dir(x):
    for root, dirs, files in os.walk(x):
        for filename in files:
            filename = os.fsdecode(filename)
            filesplit = str(filename).split('_')
            if (filename.endswith(".tar") or filename.endswith(".gz")) and validpairs.str.contains(filesplit[1]).any():
                df = untarfile(filename, root,'1')
                cur_dates = df['01']['date'].unique()
                cur_dates.sort()
                for cur_date in cur_dates:
                    cf = df.loc[df['date'] == cur_date]
                    plt.hist(cf['price'], weights=cf['amount'])
                    plt.show()
                continue
            else:
                continue



def untarfile(file, dir, day):
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


rec_check_dir(DOWNLOAD_LOCATION_PATH)