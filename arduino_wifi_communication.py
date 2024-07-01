#!/usr/bin/env python3

from socket import socket, AF_INET, SOCK_STREAM
from struct import pack

class ArduinoCommunicator:
    '''
    Class to send data to the Arduino via WiFi with a TCP connection
    '''
    def __init__(self, ip='192.168.0.21', port=50000, buffer_size=1):
        """
        Initialize the ArduinoCommunicator with IP address, port, and buffer size.
        """
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.BUFFER_SIZE = buffer_size
        self.socket = None
        self.k = 0
        self.step = 0.01

    def __del__(self):
        self.close()

    def connect(self):
        """
        Establish a TCP connection to the Arduino.
        """
        print('Waiting for Simulink to start')
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.setblocking(True)
        self.socket.connect((self.TCP_IP, self.TCP_PORT))
        print("Connection established!")

    def send_weight(self, weight):
        """
        Send the weight data to the Arduino via TCP.
        Parameters: weight (float)
        """
        msg = pack('<f', weight)
        self.socket.send(msg)
        print('Sent data:', weight)

    def close(self):
        '''
        Close TCP connection
        '''
        if self.socket:
            self.socket.close()
            print("Connection closed")