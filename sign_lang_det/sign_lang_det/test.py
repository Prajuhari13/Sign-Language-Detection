import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier=Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
folder="Data/A"
counter=0

labels=['A','B','C','E','F','I','K','L','O','R','U','V','W','Y','G','H']

while True:
    success, img = cap.read()
    imgoutput=img.copy()
    hands,img= detector.findHands(img,draw=False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Check if the bounding box dimensions are valid
        if w > 0 and h > 0:
            img_crop = img[max(0, y - offset):min(y + h + offset, img.shape[0]), 
                           max(0, x - offset):min(x + w + offset, img.shape[1])]
            
            # Resize img_crop to match the dimensions of imgwhite
            img_crop_resized = cv2.resize(img_crop, (300, 300))
            
            # Create a white image (destination image)
            imgwhite = 255 * np.ones((300, 300, 3), dtype=np.uint8)
            
            aspectRatio = h / w
            if aspectRatio > 1:
                k = 300 / h
                wcal = math.ceil(k * w)
                imgResize = cv2.resize(img_crop, (300, 300))
                imgwhite[0:imgResize.shape[0], 0:imgResize.shape[1]] = img_crop_resized
                prediction,index = classifier.getPrediction(imgwhite)
                print(prediction,index)
            
            else:
                imgResize = cv2.resize(img_crop, (300, 300))
                imgwhite[0:imgResize.shape[0], 0:imgResize.shape[1]] = img_crop_resized
                prediction,index = classifier.getPrediction(imgwhite,draw=False)

            
            cv2.putText(imgoutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
            cv2.rectangle(imgoutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)
            
        
    cv2.imshow("Image", imgoutput)
    cv2.waitKey(1)

