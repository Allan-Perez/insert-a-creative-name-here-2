#!/usr/bin/python

import json
import socket
import logging
import binascii
import struct
import argparse
import random
import time
from communication import ServerComms
from communication import ServerMessageTypes
from movement import Movement
from neuralNetwork import NeuralNetwork
from info import InformationExtraction


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
# init information extraction library
infoExtraction = InformationExtraction(gameServer)
# init movement library
movement = Movement(gameServer)
# Spawn our tank
logging.info("Creating tank with name '{}'".format(args.name))
gameServer.sendMessage(ServerMessageTypes.CREATETANK, {'Name': args.name})


# Main loop - read game messages, ignore them and randomly perform actions
def mainLoop():
    message = gameServer.readMessage()
    movement.turnTank(random.random())
    movement.move(random.random())
    movement.turnTurret(random.random())


# starter
printedFps = False
while True:
    startTime = time.clock()
    movement.stopAll()
    mainLoop()
    endTime = time.clock()
    fps = 1.0 / (endTime - startTime)

    if (int(time.clock()) & 10 and printedFps is False):
        # print every 10 seconds
        print("{} fps}", fps)
        printedFps = True
    else:
        printedFps = False
