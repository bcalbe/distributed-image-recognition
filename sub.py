#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import cv2

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.SUB)
socket.connect("tcp://192.168.141.1:5555")
socket.setsockopt(zmq.SUBSCRIBE,b"")

#  Do 10 requests, waiting each time for a response
# for request in range(10):
# print("Sending request %s …" % request)
# socket.send(b"Hello")

#  Get the reply.
image = socket.recv()
image = cv2.imdecode(image,cv2.IMREAD_COLOR)
cv2.imshow("image",image)
cv2.waitKey(0)
print("Received reply %s [ %s ]" % (request, message))