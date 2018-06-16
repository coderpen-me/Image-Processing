# import the necessary packages

import sys
import numpy as np
import argparse
import imutils
import glob
import cv2
import time
import RPi.GPIO as GPIO

intTz1 = 11
intTz2 = 13
intTz3 = 15

GPIO.setmode(GPIO.BOARD)

GPIO.setwarnings(False)
GPIO.setup(intTz1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(intTz2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(intTz3, GPIO.OUT, initial=GPIO.LOW)

j = 0
startX = 0
startY = 0
endX = 0
endY = 0
c = 100
c1 = 400
i = 0
cap = cv2.VideoCapture(0)
while(cap.isOpened() == False):
    cap.release()
    cap = cv2.VideoCapture(0)
ret, imageCam = cap.read()


while(True):

    

    if i == 0 :
        
        ret, imageCam = cap.read()

        if j == 0 :
            template = cv2.imread('image.png')
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            template = cv2.Canny(template, 50, 200)
            (tH, tW) = template.shape[:2]
            #cv2.imshow('Template', template)
            image = imageCam
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found = None
        
            for scale in np.linspace(0.2, 1.0, 20)[::-1]:

            # resize the image according to the scale, and keep track
            # of the ratio of the resizing

                resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
                r = gray.shape[1] / float(resized.shape[1])

                # if the resized image is smaller than the template, then break
                # from the loop

                if resized.shape[0] < tH or resized.shape[1] < tW:
                    break

                # detect edges in the resized, grayscale image and apply template
                # matching to find the template in the image
    
                edged = cv2.Canny(resized, 50, 200)
                result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        
                # if we have found a new maximum correlation value, then ipdate
                # the bookkeeping variable
                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)
            # unpack the bookkeeping varaible and compute the (x, y) coordinates
            # of the bounding box based on the resized ratio

            (_, maxLoc, r) = found
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH)* r))

            # draw a bounding box around the detected result and display the image
            j = 1
        
        #cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        #cv2.imwrite('Image.jpg', image)
        #

        imagebl = imageCam
        imagebl = imagebl[startY:endY+20, startX-10:endX+10]
        blur=cv2.blur(imagebl,(5,5))
        blur=cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lower = np.array([0,196,98], dtype="uint8")
        upper = np.array([12,228,255], dtype="uint8")
        thresh = cv2.inRange(blur, lower, upper)
        thresh2 = thresh.copy()
        imagebl, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        best_cnt = 1
    
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                best_cnt = cnt


        M = cv2.moments(best_cnt)
        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        #cv2.circle(blur,(cx,cy),10,(0,0,255),-1)
        #cv2.imwrite("Frameblur.jpg", blur)
        if (cx != 0 and cy != 0) :

            if (cx+startX >= startX and cx+startX <= endX) and (cy+startY >= startY and cy+startY <= endY) :
                print("Loaded")
                
                imageuse = imageCam[cy+startY:cy+c1+startY, cx-c+30+startX:cx+c+startX]
                #cv2.imwrite("roi.jpg", imageuse)

                blurS=cv2.blur(imageuse,(5,5))
                blurS=cv2.cvtColor(blurS, cv2.COLOR_BGR2HSV)
                lowerB = np.array([105,179,110], dtype="uint8")#Blue Shuttlecock
                upperB = np.array([140,255,255], dtype="uint8")
                lowerBr = np.array([10,100,20], dtype="uint8")#brown Shuttlecock
                upperBr = np.array([20,255,200], dtype="uint8")
                lowerG = np.array([20,183,158], dtype="uint8")#Golden Shuttlecock
                upperG = np.array([35,255,255], dtype="uint8")
                    
                threshB = cv2.inRange(blurS, lowerB, upperB)
                threshB2 = threshB.copy()
                #cv2.imshow("fram", threshB2)
                threshBr = cv2.inRange(blurS, lowerBr, upperBr)
                threshBr2 = threshBr.copy()
                threshG = cv2.inRange(blurS, lowerG, upperG)
                threshG2 = threshG.copy()
                    
                image_B, contours_B, hierarchy_B = cv2.findContours(threshB, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                image_Br, contours_Br, hierarchy_Br = cv2.findContours(threshBr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                image_G, contours_G, hierarchy_G = cv2.findContours(threshG, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
                max_area_B = 0
                best_cnt_B = 1
        
                max_area_Br = 0
                best_cnt_Br = 1

                max_area_G = 0
                best_cnt_G = 1
    
                for cnt_B in contours_B:
                    area_B = cv2.contourArea(cnt_B)
                    if area_B > max_area_B:
                        max_area_B = area_B
                        best_cnt_B = cnt_B
                for cnt_Br in contours_Br:
                    area_Br = cv2.contourArea(cnt_Br)
                    if area_Br > max_area_Br:
                        max_area_Br = area_Br
                        best_cnt_Br = cnt_Br
                for cnt_G in contours_G:
                    area_G = cv2.contourArea(cnt_G)
                    if area_G > max_area_G:
                        max_area_G = area_G
                        best_cnt_G = cnt_G    
                M_B = cv2.moments(best_cnt_B)
                cx_B, cy_B = int(M_B['m10']/M_B['m00']), int(M_B['m01']/M_B['m00'])
        
                M_Br = cv2.moments(best_cnt_Br)
                cx_Br, cy_Br = int(M_Br['m10']/M_Br['m00']), int(M_Br['m01']/M_Br['m00'])
    
                M_G = cv2.moments(best_cnt_G)
                cx_G, cy_G = int(M_G['m10']/M_G['m00']), int(M_G['m01']/M_G['m00'])

                if(cx_B != 0 and cy_B !=0 ):
                    GPIO.output(intTz1, GPIO.HIGH)
                    print("TZ1")
                    
                    sys.exit()
                '''if(cx_Br != 0 and cy_Br !=0 ):
                    GPIO.output(intTz2, GPIO.HIGH)
                    sys.exit()'''
                    
                    
                if(cx_G != 0 and cy_G !=0 ):
                    GPIO.output(intTz3, GPIO.HIGH)
                    print("TZ3")
                    sys.exit()
                #cv2.waitKey(0) & 0xFF
        
        else :
            print("Unloaded")
        GPIO.output(intTz1, GPIO.LOW)
        GPIO.output(intTz2, GPIO.LOW)
        GPIO.output(intTz3, GPIO.LOW)
        
        #cv2.imwrite("final.jpg", imageCam)
    
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

cap.release()
cv2.destroyAllWindows()
