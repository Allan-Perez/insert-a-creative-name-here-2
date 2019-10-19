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
        value %= 360
        logging.info("Turning {} degrees".format(value))
        self.gameServer.sendMessage(ServerMessageTypes.TURNTOHEADING, {'Amount': value})

    def turnTankLeft(self, amount, currHeading):
        c = 10
        amount *= c
        value = currHeading - amount
        self.gameServer.sendMessage(ServerMessageTypes.STOPTURN)
        self.gameServer.sendMessage(ServerMessageTypes.TURNTOHEADING, {'Amount': value})

    def turnTankRight(self, amount, currHeading):
        c = 10
        amount *= c
        value = currHeading + amount
        self.gameServer.sendMessage(ServerMessageTypes.STOPTURN)
        self.gameServer.sendMessage(ServerMessageTypes.TURNTOHEADING, {'Amount': value})

    def move(self, amount):
        #1 = c units
        c = 10 #max movement
        value = amount * c
        logging.info("Moving {} units".format(value))
        self.gameServer.sendMessage(ServerMessageTypes.MOVEFORWARDDISTANCE, {'Amount': value})

    def turnTurret(self, angle):
        value = angle*360-180
        value %= 360
        logging.info("Turning turret to {}".format(value))
        #ss
        self.gameServer.sendMessage(ServerMessageTypes.TURNTURRETTOHEADING, {'Amount': value})

    def turnTurretRight(self, amount, currTurHeading):
        c = 10
        amount *= c
        value = currTurHeading + amount
        self.gameServer.sendMessage(ServerMessageTypes.STOPTURN)
        self.gameServer.sendMessage(ServerMessageTypes.TURNTURRETTOHEADING, {'Amount': value})

    def turnTurretLeft(self, amount, currTurHeading):
        c = 10
        amount *= c
        value = currTurHeading - amount
        self.gameServer.sendMessage(ServerMessageTypes.STOPTURN)
        self.gameServer.sendMessage(ServerMessageTypes.TURNTURRETTOHEADING, {'Amount': value})

    def stopAll(self):
        #stops all movement and turning of tank
        #usually called each frame
        self.gameServer.sendMessage(ServerMessageTypes.STOPALL)

    def fire(self):
        self.gameServer.sendMessage(ServerMessageTypes.FIRE)

