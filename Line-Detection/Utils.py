import numpy as np
import cv2
import time
from Image import *



def SlicePart(im, images, slices):
    height, width = im.shape[:2]
    sl = int(height/slices);
    values = [None] * 4
    for i in range(slices):
        part = sl*i
        crop_img = im[part:part+sl, 0:width]
        images[i].image = crop_img
        values[i] = images[i].Process()
    return (values[0], values[1], values[2], values[3])
    
def RepackImages(images):
    img = images[0].image
    for i in range(len(images)):
        if i == 0:
            img = np.concatenate((img, images[1].image), axis=0)
        if i > 1:
            img = np.concatenate((img, images[i].image), axis=0)
            
    return img

def Center(moments):
    if moments["m00"] == 0:
        return 0
        
    x = int(moments["m10"]/moments["m00"])
    y = int(moments["m01"]/moments["m00"])

    return x, y
    
def RemoveBackground(image, b):
    up = 100
    # create NumPy arrays from the boundaries

    
    blur=cv2.GaussianBlur(image,(5,5), 0)
    blur=cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    lower = np.array([0,0,0], dtype="uint8")
    upper = np.array([179,255,90], dtype="uint8")
    
    #----------------COLOR SELECTION-------------- (Remove any area that is whiter than 'upper')
    if b == True:
        mask = cv2.inRange(image, lower, upper)
        image = cv2.bitwise_and(image, image, mask = mask)
        image = cv2.bitwise_not(image, image, mask = mask)
        image = (255-image)
        return image
    else:
        return image
    #////////////////COLOR SELECTION/////////////
    
