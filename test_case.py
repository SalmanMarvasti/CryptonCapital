#import modellingmanager
from modellingmanager import prediction_checker
#from modellingmanager import reload
import time
import importlib
# import test
def test_case():
    p= prediction_checker()
    timestamp = time.time()
    p.update(100, timestamp, 0.5)
    p.add_pred(timestamp+2000, 101,1)
    p.update(101, timestamp, 0.5)
    print(p.number_correct==1)



#importlib.reload(modellingmanager)
test_case()