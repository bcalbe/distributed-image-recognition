import time
import zmq
import socket as soc
import cv2


ip = soc.gethostbyname(soc.gethostname())
print(ip)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    #  Wait for next request from client
    message = socket.recv()
    print("Received request: {}".format(message))

    #  Do some 'work'
    time.sleep(1)

    #  Send reply back to client
    socket.send(b"World")

context.destroy()
