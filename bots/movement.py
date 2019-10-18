from communication import ServerComms
from communication import ServerMessageTypes

import logging
class Movement:
    gameServer : ServerComms = None
    def __init__(self, gameServer):
        self.gameServer = gameServer

    def turnTank(self, amount):
        #0.5 == 0 degrees, 1 == 180 degrees, 0.25 = -90 degrees        
        value = amount * 360 - 180
        logging.info("Turning {} degrees".format(value))
        self.gameServer.sendMessage(ServerMessageTypes.TURNTOHEADING, {'Amount': value})

    def move(self, amount):
        #1 = c units
        c = 10 #max movement        
        value = amount * c
        logging.info("Moving {} units".format(value))
        self.gameServer.sendMessage(ServerMessageTypes.MOVEFORWARDDISTANCE, {'Amount': value})

