
from responsemanager import credentials
from responsemanager import PublishServer

credentials.api = "1f28b5ccdec84a30bbd0231bb210c7d7"
credentials.secret = "Test@123"
credentials.endpoint = "https://api.coinigy.com/api/v2"
cr = PublishServer(credentials)
cr.read()  #'tradesizerequest'

