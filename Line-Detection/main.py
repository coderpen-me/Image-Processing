import socket
import sys
import os
import numpy as np
import pdb

import cv2
import time

from Image import *
from Utils import *

font = cv2.FONT_HERSHEY_SIMPLEX
direction = 0
Images=[]
N_SLICES = 4

flag = 0

for q in range(N_SLICES):
    Images.append(Image())

cope = cv2.VideoCapture(0)

while True:
    ret, img = cope.read()
    direction = 0
    blur=cv2.blur(img,(5,5))
    blur=cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,0], dtype="uint8")
    upper = np.array([179,255,121], dtype="uint8")
    
    thresh = cv2.inRange(blur, lower, upper)

    cv2.imshow("IMG", thresh)
    
    imagebl, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_cnt = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt


    M = cv2.moments(best_cnt)

    if( flag == 0):

        print("START")
        if(M['m00'] > 110000):
            flag = 0
        else:
            flag = 1

    if( flag == 1):        

        if(M['m00'] > 110000):
            flag = 2
        else:
            flag = 1
        
        img = RemoveBackground(img, False)
        if img is not None:


            values = [None] * 5
             
            t1 = time.clock()
            values[0], values[1], values[2], values[3] = SlicePart(img, Images, N_SLICES)

            if (values[0]-values[3]>40):
                print("Rotate AnticlockWise")
            if (values[0]-values[3]<-40):
                print("Rotate clockWise")

            if ( (values[0] > 30)and (values[2] > 30) and (values[1] > 30) and (values[3] > 30)) :
                print("Go LEFT")
            if ( (values[0] < -30)and (values[2] < -30) and (values[1] < -30) and (values[3] < -30)) :
                print("Go RIGHT")
            
            if ( (values[0] < 30)and (values[2] < 30) and (values[1] < 30) and (values[3] < 30) and (values[0] > -30)and (values[2] > -30) and (values[1] > -30) and (values[3] > -30)) :
                print("Go Straight")
            
            for i in range(N_SLICES):
                fm = RepackImages(Images)
                t2 = time.clock()
                cv2.putText(fm,"Time: " + str((t2-t1)*1000) + " ms",(10, 470), font, 0.5,(0,0,255),1,cv2.LINE_AA)
                cv2.imshow("Vision Race", fm)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    if ( flag == 2):

        print("STOP")

        



cv2.destroyAllWindows()
connection.close()
