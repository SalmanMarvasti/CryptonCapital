from twisted.internet import protocol
from twisted.internet import reactor
import pandas as pd
import os
DOWNLOAD_LOCATION_PATH = os.path.expanduser("~") + "/s3-backup/"

class Writer(protocol.ProcessProtocol):
    def __init__(self, data):
        self.data = data

    def connectionMade(self):
        print("Writer -- connection made")
        self.transport.writeToChild(0, self.data)
        self.transport.closeChildFD(0)

    def childDataReceived(self, fd, data):
        pass

    def processEnded(self, status):
        pass


class Reader(protocol.ProcessProtocol):
    def __init__(self):

        self.received = ''
    pass

    def connectionMade(self):
        print("Reader -- connection made")
        pass


    def childDataReceived(self, fd, data):
        print("Reader -- childDataReceived")
        self.received = data


    def processEnded(self, status):
        print("process ended, got:", self.received)


class WriteRead(protocol.ProcessProtocol):
    def __init__(self, data):
        self.data = data

    def connectionMade(self):
        self.transport.writeToChild(0, self.data)
        self.transport.closeChildFD(0)

    def childDataReceived(self, fd, data):
        self.received = data
        print("got data:", data)

    def processEnded(self, status):
        print("process ended - now what?")

def test1(data):
    # just call wc
    p2 = reactor.spawnProcess(WriteRead(data), "wc", ["wc"], env=None, childFDs={0: "w", 1: "r"})
    reactor.run()

def test2(data):
    rfd, wfd = os.pipe()
    cargs = ["cat"]
    wargs = ["wc", "-w"]
    p1 = reactor.spawnProcess(Writer(data), "cat",cargs , env=None, childFDs={0:"w", 1: wfd })
    p2 = reactor.spawnProcess(Reader(),     "wc",wargs, env=None, childFDs={0: rfd, 1: "r"})
    os.close(rfd)
    os.close(wfd)
    reactor.run()

if __name__ == "__main__":
    #test2('this is a test'.encode('utf-8'))
    df = pd.read_csv(DOWNLOAD_LOCATION_PATH + 'binance.csv')
    validpairs = df['PairNameApi']
    for pair in validpairs:
        wargs = ["python", ""]
        p2 = reactor.spawnProcess(Reader(), "python modellingmanager.py", wargs, env=None)

    reactor.run()