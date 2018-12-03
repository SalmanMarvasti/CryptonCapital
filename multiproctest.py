import multiprocessing as mp
from random import uniform, randrange

def flop_no(rand_nos, a, b):
    cals = []
    for r in rand_nos:
        cals.append(r + a * b)
    return cals


def flop(val, a, b, out_queue):
    cals = []
    for v in val:
        cals.append(v + a * b)
    out_queue.put(cals)
    # time.sleep(3)


def concurrency():
    out_queue = mp.Queue()
    a = 3.3
    b = 4.4
    rand_nos = [uniform(1, 4) for i in range(1000000)]
    print  (len(rand_nos))
    # for i in range(5):
    start_time = time.time()
    p1 = mp.Process(target=flop, args=(rand_nos[:250000], a, b, out_queue))
    p2 = mp.Process(target=flop, args=(rand_nos[250000:500000], a, b, out_queue))
    p3 = mp.Process(target=flop, args=(rand_nos[500000:750000], a, b, out_queue))
    p4 = mp.Process(target=flop, args=(rand_nos[750000:], a, b, out_queue))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    print(len(out_queue.get()))
    print(out_queue.get())
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    print("Running time parallel: ", time.time() - start_time, "secs")


def no_concurrency():
    a = 3.3
    b = 4.4
    rand_nos = [uniform(1, 4) for i in range(1000000)]
    start_time = time.time()
    cals = flop_no(rand_nos, a, b)
    print
    "Running time serial: ", time.time() - start_time, "secs"

