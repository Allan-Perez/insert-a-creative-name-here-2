from communication import ServerComms
from communication import ServerMessageTypes

import logging

gameObjects = []


class InformationExtraction:
    gameServer : ServerComms = None

    def __init__(self, gameServer):
        self.gameServer = gameServer

    def readObjectUpdate(self):
        message = self.gameServer.readMessage()

        if message['messageType'] == 18:

            # if the array is empty append first game object
            if len(gameObjects) == 0:
                gameObjects.append(message)

            else:
                used = False
                for i,dict in enumerate(gameObjects):
                    if dict['Id'] == message['Id']:
                        # update element
                        gameObjects[i] = message
                        used = True
                        break
                if not used:
                    gameObjects.append(message)



        print(message, type(message))



class GameObject:

    Id = None
    Name = None
    Type = None
    X = None
    Y = None
    Heading = None
    TurretHeading = None
    Health = None
    Ammo = None


    def __init__self(self, dict):
        self.Id = dict['Id']
        self.Name = dict['Name']
        self.Type = dict['Type']
        self.X = dict['X']
        self.Y = dict['Y']
        self.Heading = dict['Heading']
        self.TurretHeading = dict['TurretHeading']
        self.Health = dict['Health']
        self.Ammo = dict['Ammo']

    def getId(self):
        return self.Id

    def getName(self):
        return self.Name

    def getType(self):
        return self.Type

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def getCoords(self):
        return (self.X, self.Y)

    def getHeading(self):
        return self.Heading

    def getTurretHeading(self):
        return self.TurretHeading

    def getHealth(self):
        return self.Health

    def getAmmo(self):
        return self.Ammo

    def setX(self, x):
        self.X = x

    def setY(self, y):
        self.Y = y

    def setHeading(self, heading):
        self.Heading = heading

    def setTurretHeading(self, turretHeading):
        self.TurretHeading = turretHeading

    def setHealth(self, health):
        self.Health = health

    def setAmmo(self, ammo):
        self.Ammo = ammo






