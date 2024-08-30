import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20

folder="Data/Z"
counter=0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

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
            else:
                imgResize = cv2.resize(img_crop, (300, 300))
                imgwhite[0:imgResize.shape[0], 0:imgResize.shape[1]] = img_crop_resized

            # Display the image with the cropped region copied onto a white background
            cv2.imshow("imageWhite", imgwhite)
            # cv2.imshow('imagecrop', img_crop)
        
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgwhite)
        print(counter)

