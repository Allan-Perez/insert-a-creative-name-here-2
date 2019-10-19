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

    def wallDistance(posX, posY, heading, forward=True):
        def further_point(oldx, oldy, angle):
            angle = angle % 360
            x, y = oldx, oldy
            if angle == 0: #angle increases in mathematically negative direction
                x = oldx + 10
            elif angle == 90:
                y = oldy + 10
            elif angle == 180:
                x = oldx - 10
            elif angle == 270:
                y = oldy - 10
            elif 0 < angle < 180:
                y = oldy + 10
                x = oldx + 10 * math.tan(angle)
            else:
                y = oldy - 10
                x = oldx - 10 * math.tan(angle)
                
        print("x, y =", x, y)
        return x, y

        def line_intersection(line1, line2):
            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        #print("vnutri", line1, line2, div)
        if div == 0:
            return "none"

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

        closest_distance = 100000000000
    
        if forward:
            a = "forward"
        else:
            a = "backward"
            heading = heading + 180
        logging.info("Calculating closest wall in {} direction".format(a))
        current_pos = (posX, posY)
        future_pos = further_point(posX, posY, heading)
        line = (current_pos, future_pos)
        for i in range((len(playground_points))):
            if i == 0:
                pl_line = (playground_points[i], playground_points[-1])
            else:
                pl_line = (playground_points[i], playground_points[i-1])
            intersect_pt = line_intersection(line, pl_line)
            if intersect_pt == "none":
                continue
            if ((pl_line[0][0] <= intersect_pt[0] <= pl_line[1][0] or pl_line[0][0] >= intersect_pt[0] >= pl_line[1][0])
                    and (pl_line[0][1] <= intersect_pt[1] <= pl_line[1][1] or pl_line[0][1] >= intersect_pt[1] >= pl_line[1][1])):
                distance = math.sqrt((intersect_pt[0]-posX)**2 + (intersect_pt[1]-posY)**2)
                if distance < closest_distance:
                    closest_distance = distance
                    print("INTERSECTION", intersect_pt, pl_line)
            else:
                continue
        return closest_distance



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






