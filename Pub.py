import time
import zmq
import socket as soc
import cv2

ip = soc.gethostbyname(soc.gethostname())
print(ip)

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")
image = cv2.imread("./video/image/2.0_0_airplane.jpg")
image = cv2.imencode(".jpg",image)[1].tobytes()

while True:
    #  Wait for next request from client
    # message = socket.recv()
    # print("Received request: %s" % message)

    #  Do some 'work'
    #time.sleep(1)

    #  Send reply back to client
    socket.send(image)


    