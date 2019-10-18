#!/usr/bin/python

import json
import socket
import logging
import binascii
import struct
import argparse
import random

from communication import ServerComms
from communication import ServerMessageTypes
from movement import Movement

# Parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
parser.add_argument('-H', '--hostname', default='127.0.0.1', help='Hostname to connect to')
parser.add_argument('-p', '--port', default=8052, type=int, help='Port to connect to')
parser.add_argument('-n', '--name', default='TeamA:RandomBot', help='Name of bot')
args = parser.parse_args()

# Set up console logging
if args.debug:
	logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.DEBUG)
else:
	logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)


# Connect to game server
gameServer = ServerComms(args.hostname, args.port)
#init movement library
movement = Movement(gameServer)
# Spawn our tank
logging.info("Creating tank with name '{}'".format(args.name))
gameServer.sendMessage(ServerMessageTypes.CREATETANK, {'Name': args.name})

# Main loop - read game messages, ignore them and randomly perform actions
i=0
while True:
	message = gameServer.readMessage()
    
	if i == 5:
		if random.randint(0, 10) > 5:
			logging.info("Firing")
			gameServer.sendMessage(ServerMessageTypes.FIRE)
	elif i == 10:
		movement.moveRight()
	elif i == 15:
		logging.info("Moving randomly")
		gameServer.sendMessage(ServerMessageTypes.MOVEFORWARDDISTANCE, {'Amount': random.randint(0, 10)})
	movement.moveRight()
	i = i + 1
	if i > 20:
		i = 0

