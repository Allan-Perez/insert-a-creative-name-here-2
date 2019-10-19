from communication import ServerComms
from communication import ServerMessageTypes
from bots import bot

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

    def turnTankLeft(self, amount):
        currentHeading = None

        gameObjects = bot.infoExtraction.gameObjects
        botArgsName = bot.args.name
        # if the object's name is identical to the name of the bot created
        for object in gameObjects:
            if object['Name'] == botArgsName:
                # set its current heading to the value from dict
                currentHeading = object['Heading']
                break
        c = 10
        change = (amount * c) % 360
        value = currentHeading + change
        self.gameServer.sendMessage(ServerMessageTypes.TURNTOHEADING, {'Amount': value})


    def turnTankRight(self, amount):
        currentHeading = None

        gameObjects = bot.infoExtraction.gameObjects
        botArgsName = bot.args.name
        # if the object's name is identical to the name of the bot created
        for object in gameObjects:
            if object['Name'] == botArgsName:
                # set its current heading to the value from dict
                currentHeading = object['Heading']
                break
        c = 10
        change = (amount * c) % 360
        value = currentHeading + change
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

    def turnTurretLeft(self, amount):
        currentTurrHeading = None
        # if the object's name is identical to the name of the bot created
        gameObjects = bot.infoExtraction.gameObjects
        botArgsName = bot.args.name

        for object in gameObjects:
            if object['Name'] == botArgsName:
                # set its current heading to the value from dict
                currentTurrHeading = object['TurretHeading']
                break
        c = 10
        change = (amount * c) % 360
        value = currentTurrHeading + change
        self.gameServer.sendMessage(ServerMessageTypes.TURNTURRETTOHEADING, {'Amount': value})


    def turnTurretRight(self, amount):
        currentTurrHeading = None
        # if the object's name is identical to the name of the bot created
        gameObjects = bot.infoExtraction.gameObjects
        botArgsName = bot.args.name

        for object in gameObjects:
            if object['Name'] == botArgsName:
                # set its current heading to the value from dict
                currentTurrHeading = object['Heading']
                break
        c = 10
        change = (amount * c) % 360
        value = currentTurrHeading + change
        self.gameServer.sendMessage(ServerMessageTypes.TURNTURRETTOHEADING, {'Amount': value})



    def stopAll(self):
        #stops all movement and turning of tank
        #usually called each frame
        self.gameServer.sendMessage(ServerMessageTypes.STOPALL)

