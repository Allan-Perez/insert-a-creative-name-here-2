from communication import ServerComms
from communication import ServerMessageTypes

import random
import logging
class Movement:
    gameServer : ServerComms = None
    def __init__(self, gameServer):
        self.gameServer = gameServer

    def moveRight(self):
        logging.info("Turning randomly")
        #ss
        self.gameServer.sendMessage(ServerMessageTypes.TURNTOHEADING, {'Amount': random.randint(0, 359)})
        #self.gameServer.sendMessage(ServerMessageTypes.FIRE)

