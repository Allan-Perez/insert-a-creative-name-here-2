from communication import ServerComms
from communication import ServerMessageTypes
import numpy as np 

import logging



class InformationExtraction:
	gameObjects = []
	convertedGameObjects = np.zeros(119)
	gameServer : ServerComms = None

	def __init__(self, gameServer):
		self.gameServer = gameServer

	def readObjectUpdate(self):
		message = self.gameServer.readMessage()
		#18 means that it is a game object
		if message['messageType'] == 18:

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
		for i in range(4):
			if convDict[0] == self.convertedGameObjects[index + 7*i]:				
				self.convertedGameObjects[index + i*7: index + (i + 1) * 7] = convDict
				break
			if self.convertedGameObjects[index + i * 7] == 0:
				self.convertedGameObjects[index + i*7: index + (i + 1) * 7] = convDict
				break    	
	def convertGameObjects(self):
		for gameObject in self.gameObjects:
			convertedDictionary = []

			for key in gameObject.keys():
				if key == 'Id':
					convertedDictionary.append( 1/ (-gameObject[key])) #just ids
				elif key == 'X':    					
					convertedDictionary.append((gameObject[key] + 70) / 140) #top is 1 and bottom is zero
				elif key == 'Y':    					
					convertedDictionary.append((gameObject[key] + 100) / 200) #left is 1 and right is zero
				elif key == 'Heading':    					
					convertedDictionary.append(((gameObject[key]/360)+0.5)%1) #0.5 == 0 degrees, 1 == 180 degrees, 0.25 = -90 degrees     
				elif key == 'TurretHeading':  					
					convertedDictionary.append(((gameObject[key]/360)+0.5)%1) #0.5 == 0 degrees, 1 == 180 degrees, 0.25 = -90 degrees     		
				elif key == 'Health':
					convertedDictionary.append(gameObject[key]/10)  					
				elif key == 'Ammo':
					convertedDictionary.append(gameObject[key]/10)  
			if gameObject['Type'] == 'Tank':
				if gameObject['Name'] == 'OurTeam:Bots':
					self.pasteObjectToConvertedGameObjects(convertedDictionary)
				else: #enemy tanks
					self.pasteObjectToConvertedGameObjects(convertedDictionary,28)
			elif gameObject['Type'] == 'AmmoPickup':
				self.pasteObjectToConvertedGameObjects(convertedDictionary,56)
			elif gameObject['Type'] == 'HealthPickup':
				self.pasteObjectToConvertedGameObjects(convertedDictionary,84)
			else:				
				for i in range(7):
					self.convertedGameObjects[112:] = convertedDictionary