#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import time
import zmq
import socket as soc 


ip = soc.gethostbyname(soc.gethostname())
print(ip)


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://192.168.141.1:5556")

while True:
    #  Wait for next request from client
    message = socket.recv()
    print("Received request: %s" % message)

    #  Do some 'work'
    time.sleep(1)

    #  Send reply back to client
    socket.send(b"World")
context.destroy()