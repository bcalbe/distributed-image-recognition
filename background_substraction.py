import cv2
import numpy as np
import tensorflow as tf
import network
    

def diff(prevframe,curframe):
    mask = cv2.absdiff(prevframe,curframe)
    flag,mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
    return mask

def inference(ROI):
    
    pass

classes = ["airplane","automobile", "bird","cat", "deer", "dog","frog","horse", "ship", "truck"]
nums_classes = [0,0,0,0,0,0,0,0,0,0]
capture = cv2.VideoCapture("./video/Town Square_1.mp4")

Serial_model = network.Get_SerialModel()
model  = tf.keras.models.load_model("./model/VGG19_11.h5")

success,prevframe = capture.read()
prevframe = cv2.cvtColor(prevframe,cv2.COLOR_BGR2GRAY)
prevframe = cv2.resize(prevframe,(640,480))
prevframe = cv2.GaussianBlur(prevframe, (5, 5), 1)
success,curframe = capture.read()
cv2.namedWindow("original video")
backsub = cv2.createBackgroundSubtractorMOG2()
kernel = np.ones((3,3),np.uint8) 

while(True):
    curframe = cv2.resize(curframe,(640,480))
    curframe_grey = cv2.cvtColor(curframe,cv2.COLOR_BGR2GRAY)
    curframe_grey = cv2.GaussianBlur(curframe_grey, (5, 5), 1)
    fgMask = backsub.apply(curframe_grey)
    #fgMask = diff(prevframe,curframe)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    fgMask, contours, hierarchy = cv2.findContours(fgMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
    detect_frame = curframe.copy()
    save = 0
    for n,c in enumerate(contours):
        if cv2.contourArea(c) < 300: # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
            continue
        (x, y, w, h) = cv2.boundingRect(c) # 该函数计算矩形的边界框
        crop = detect_frame[y:y+h,x:x+w]
        cv2.rectangle(curframe, (x, y), (x+10+w, y+10+h), (0, 255, 0), 2)
        crop = cv2.resize(crop,(32,32))
        outputs = network.run_SerialModel(Serial_model,crop[np.newaxis,:,:,:])
        #outputs = model.predict(crop[np.newaxis,:,:,:])
        max_scores = np.max(outputs)
        if max_scores >0.7:
            cv2.imwrite("./video/keyframe/{}.jpg".format(str(capture.get(cv2.CAP_PROP_POS_FRAMES))),curframe)
            object_class = np.argmax(np.array(outputs))
            nums_classes[object_class] += 1
            cv2.imwrite("./video/image/{}_{}_{}.jpg".format(str(capture.get(cv2.CAP_PROP_POS_FRAMES)),n,classes[object_class]),crop) 

    cv2.rectangle(curframe, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(curframe, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    cv2.imshow("original video",curframe)
    cv2.imshow("backsub",fgMask)
    if(cv2.waitKey(20) == ord('q')):
        print (nums_classes)
        break
    prevframe = curframe.copy()
    success,curframe = capture.read() 
    