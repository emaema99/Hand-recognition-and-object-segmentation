#!/usr/bin/env python3

####################################################################################################
# Script to send data to the Arduino via WiFi with a TCP connection
####################################################################################################

import socket
import struct
import time
import math

class ArduinoCommunicator:
    def __init__(self, ip='192.168.0.21', port=50000, buffer_size=1):
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.BUFFER_SIZE = buffer_size
        self.socket = None
        self.k = 0
        self.step = 0.01

    def __del__(self):
        self.close()

    def connect(self):
        print('Waiting for Simulink to start')
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setblocking(True)
        self.socket.connect((self.TCP_IP, self.TCP_PORT))
        print("Connection established!")

    def send_weight(self, weight):
        msg = struct.pack('<f', weight)
        self.socket.send(msg)
        print('Sent data:', weight)

    def close(self):
        if self.socket:
            self.socket.close()
            print("Connection closed")