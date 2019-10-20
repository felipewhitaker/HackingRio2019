import os
import time
import requests
import numpy as np

import cv2
from pyzbar import pyzbar

class BRSorria:

    def __init__(self, faceAPI, payAPI, prices = None):
        self._payAPI = payAPI
        self._faceAPI = faceAPI
        self._prices  = prices

        # files with the OpenCv weights for the neural network that check faces and smiles
        self._faceCascade  = cv2.CascadeClassifier('cv2_cascades/haarcascade_frontalface_default.xml')
        self._smileCascade = cv2.CascadeClassifier('cv2_cascades/haarcascade_smile.xml')

    def write_contrast(frame, x, y, text,  w = 0, h = 0, FONT = cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE = 1, FONT_THICKNESS = 2):
        """ creates a white box with black text with user instructions """
        (label_width, label_height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)

        frame = cv2.rectangle(frame, (x - (label_width - w)//2, y - 20 + 10), (x - (label_width - w)//2 + label_width, y - 20 - label_height - 10), (255, 255, 255), -1)
        frame = cv2.putText(frame, text, (x - (label_width - w)//2, y - 20), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)

        return frame

    def detect_face(self, gray, frame):
        face = self._faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces: # check for faces
            cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

        return frame, roi_gray, roi_color

    def detect_smile(self, gray, frame):
        smiles = self._smileCascade.detectMultiScale(gray, 1.8, 20)

        for (sx, sy, sw, sh) in smiles: # only checks a smiles within the constraint of a face
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)

            return frame, len(smiles) > 0
        return frame, False

    def execute_pay(self):
        self._payAPI()
        return

FONT_SCALE = .4 # font to print the transaction informations

qr = False # boolean to check if there is a QR Code within the video
already_paid = False # if qr code had the transaction Id, we could check if it was already paid with the post request

resol_array = (1024, 720) # camera resolution ~ lower is easier for the computer
minY = resol_array[1]//12

faceAPI = {'Felipe': '12 pães de queijo'}
payAPI  = None
prices  = {'12 pães de queijo': 9.99} 

BR = BRSorria(faceAPI, payAPI, prices)

# start capturing video
cap = cv2.VideoCapture(0)
start = time.time()
while(True):
    ret, frame = cap.read()
    # resize the frame so we can save computer energy
    frame = cv2.resize(frame, resol_array)

    # checks if the frame was correctly processed
    if ret:
        # creates a gray version of the frame to facilitate processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if time.time() - start > 1/120:
            frame, smile = BR.detect_smile(gray, frame)
        else:
            start = time.time()

        person = 'Felipe'
        recommendation = BR._faceAPI[person]
        price = BR._prices[recommendation]

        if recommendation:
            frame = BR.write_contrast(frame, *resol_array, f'[GOSTARIA DE COMPRAR {recommendation.upper()} por RS{round(price, 2)}]?') 

        # Maintain the success message
        if smile and not already_paid:
            already_paid = smile
            # BR.execute_pay() - No API was given!

        # write help messages for the user - either success or smile to pay
        if already_paid:
            frame = BR.write_contrast(frame, resol_array[0]//2, minY, f'[SUCESSO]')
        else:
            frame = BR.write_contrast(frame, resol_array[0]//2, minY, f'[SORRIA PARA PAGAR]')

        cv2.imshow('frame', frame)

        # quits if the Q key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # quits the program if the image cannot be processed correctly
    else:
        break

# destroys OpenCV window
cv2.destroyAllWindows()
