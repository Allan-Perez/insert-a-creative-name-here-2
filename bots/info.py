from communication import ServerComms
from communication import ServerMessageTypes

import logging

class InformationExtraction:
	gameServer : ServerComms = None
	def __init__(self, gameServer):
		self.gameServer = gameServer

	def readObjectUpdate(self):
		message = self.gameServer.readMessage()
		print(message, type(message))
