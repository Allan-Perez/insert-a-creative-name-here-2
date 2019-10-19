from communication import ServerComms
from communication import ServerMessageTypes
import numpy as np
import logging
import time
import math

playground_points = [(70, 70), (70, -70), (40, -100), (40, -120), (-40, -120), (-40, -100), (-70, -70), (-70, 70),
                     (-40, 70), (-40, 120), (40, 120), (40, 70)]
class MyTank:
    id = None
    name = None
    x = None
    y = None
    heading = None
    turretHeading = None

    def __init__(self, dict):
        self.id = dict['Id']
        self.name = dict['Name']
        self.x = dict['X']
        self.y = dict['Y']
        self.heading = dict['Heading']
        self.turretHeading = dict['TurretHeading']

    def getHeading(self):
        return self.heading

    def getTurretHeading(self):
        return self.turretHeading

    def getId(self):
        return self.id

    def setHeading(self, heading):
        self.heading = heading

    def setTurretHeading(self, turretheading):
        self.turretHeading = turretheading

    def setX(self, x):
        self.x = x

    def setY(self, y):
        self.y = y

    def updateInternalState(self, dict):
        self.setHeading(dict['Heading'])
        self.setTurretHeading(dict['TurretHeading'])
        self.setX(dict['X'])
        self.setY(dict['Y'])

class InformationExtraction:
	gameObjects = []
	convertedGameObjects = np.zeros(119)
	dictionaryIndex = {} #dictionary of indexes
	gameServer : ServerComms = None

	def __init__(self, gameServer):
		self.gameServer = gameServer
		self.obs_dim = convertedGameObjects.shape
		

	def getAllInfo(self):
		self.readObjectUpdate()
		self.convertGameObjects()
		return self.convertedGameObjects

	def readObjectUpdate(self):
		message = self.gameServer.readMessage()
		#18 means that it is a game object
		if message['messageType'] == 18:
			message['TimeStamp'] = time.clock()
			# if the array is empty append first game object
			if len(self.gameObjects) == 0:
				self.gameObjects.append(message)

			else:
				used = False
				for i,dict in enumerate(self.gameObjects):
					if dict['Id'] == message['Id']:
						# update element
						self.gameObjects[i] = message
						used = True
						break
				if not used:
					self.gameObjects.append(message)
	def pasteObjectToConvertedGameObjects(self, convDict, index=0):
		# paste the object into the empty spaces
		index *= 7
		self.convertedGameObjects[index: index + 7] = convDict

	def getTimeValue(self, time, c=0.4):
		return 1 / (1 + math.e**(-2 + (9 * time) / c))

	def convertGameObjects(self):
		for gameObject in self.gameObjects:
			convertedDictionary = []
			idOfObject = 0
			for key in gameObject.keys():
				if key == 'Id':
					idOfObject = gameObject[key]
				elif key == 'X':    					
					convertedDictionary.append((gameObject[key] + 80) / 160) #top is 1 and bottom is zero
				elif key == 'Y':    					
					convertedDictionary.append((gameObject[key] + 115) / 230) #left is 1 and right is zero
				elif key == 'Heading':    					
					convertedDictionary.append(((gameObject[key]/360)+0.5)%1) #0.5 == 0 degrees, 1 == 180 degrees, 0.25 = -90 degrees     
				elif key == 'TurretHeading':  					
					convertedDictionary.append(((gameObject[key]/360)+0.5)%1) #0.5 == 0 degrees, 1 == 180 degrees, 0.25 = -90 degrees     		
				elif key == 'Health':
					convertedDictionary.append(gameObject[key]/10)  					
				elif key == 'Ammo':
					convertedDictionary.append(gameObject[key]/10)
				elif key == 'TimeStamp':
					timeToPrint = self.getTimeValue(time.clock() - gameObject[key])
					convertedDictionary.append(timeToPrint)
					print(gameObject[key])
			key = idOfObject
			if key in self.dictionaryIndex:				
				self.pasteObjectToConvertedGameObjects(convertedDictionary,self.dictionaryIndex[key])
			elif gameObject['Type'] == 'Tank':
				if gameObject['Name'] == 'TeamA:1':
					self.dictionaryIndex[key] = 0
					self.pasteObjectToConvertedGameObjects(convertedDictionary,0)
				elif gameObject['Name'] == 'TeamA:2':
					self.dictionaryIndex[key] = 1
					self.pasteObjectToConvertedGameObjects(convertedDictionary,1)
				elif gameObject['Name'] == 'TeamA:3':
					self.dictionaryIndex[key] = 2
					self.pasteObjectToConvertedGameObjects(convertedDictionary,2)
				elif gameObject['Name'] == 'TeamA:4':
					self.dictionaryIndex[key] = 3
					self.pasteObjectToConvertedGameObjects(convertedDictionary,3)
				else:
					for i in range(4):
						index = 4+i
						if not index in self.dictionaryIndex.values():
							self.dictionaryIndex[key] = index
							self.pasteObjectToConvertedGameObjects(convertedDictionary,index)
							break
			elif gameObject['Type'] == 'AmmoPickup':
				for i in range(4):
					index = 8+i
					if not index in self.dictionaryIndex.values():
						self.dictionaryIndex[key] = index
						self.pasteObjectToConvertedGameObjects(convertedDictionary,index)
						break
			elif gameObject['Type'] == 'HealthPickup':
				for i in range(4):
					index = 12 + i
					if not index in self.dictionaryIndex.values():
						self.dictionaryIndex[key] = index
						self.pasteObjectToConvertedGameObjects(convertedDictionary,index)
						break
			else:
				self.dictionaryIndex[key] = 16
				self.pasteObjectToConvertedGameObjects(convertedDictionary,16)

			#if gameObject['Type'] == 'Tank':
			#	if gameObject['Name'] == 'TeamA:1':
			#		self.ourTeam(convertedDictionary,0)
			#	if gameObject['Name'] == 'TeamA:2':
			#		self.ourTeam(convertedDictionary,7)
			#	if gameObject['Name'] == 'TeamA:3':
			#		self.ourTeam(convertedDictionary,14)
			#	if gameObject['Name'] == 'TeamA:4':
			#		self.ourTeam(convertedDictionary,21)
			#	else: #enemy tanks
			#		self.pasteObjectToConvertedGameObjects(convertedDictionary,28)
			#elif gameObject['Type'] == 'AmmoPickup':
			#	self.pasteObjectToConvertedGameObjects(convertedDictionary,56)
			#elif gameObject['Type'] == 'HealthPickup':
			#	self.pasteObjectToConvertedGameObjects(convertedDictionary,84)
			#else:				
			#	for i in range(7):
			#		self.convertedGameObjects[112:] = convertedDictionary
	def wallDistance(self, posX, posY, heading, forward=True):
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
			return x, y        
		def line_intersection(line1, line2):
			xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
			ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])        
			def det(a, b):
	   			return a[0] * b[1] - a[1] * b[0]        
			div = det(xdiff, ydiff)
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
			if ((pl_line[0][0] <= intersect_pt[0] <= pl_line[1][0] or pl_line[0][0] >= intersect_pt[0] >= pl_line[1][0]) and (pl_line[0][1] <= intersect_pt[1] <= pl_line[1][1] or pl_line[0][1] >= intersect_pt[1] >= pl_line[1][1])):
				distance = math.sqrt((intersect_pt[0]-posX)**2 + (intersect_pt[1]-posY)**2)
				if distance < closest_distance:
					closest_distance = distance
					print("INTERSECTION", intersect_pt, pl_line)
			else:
				continue
		return closest_distance
