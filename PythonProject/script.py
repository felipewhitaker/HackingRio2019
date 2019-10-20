import os
import time
import requests
import numpy as np

import cv2
from pyzbar import pyzbar

def detect_smile(gray, frame):
    """ returns the modified frame that has face and smile boxes and a boolean that says if any smile was identified """
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces: # check for faces
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        for (sx, sy, sw, sh) in smiles: # only checks a smiles within the constraint of a face
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)

        return frame, len(smiles) > 0
    return frame, False

def write_contrast(frame, x, y, text,  w = 0, h = 0, FONT = cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE = 1, FONT_THICKNESS = 2):
    """ creates a white box with black text with user instructions """
    (label_width, label_height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)

    frame = cv2.rectangle(
        frame,
        (x - (label_width - w)//2, y - 20 + 10),
        (x - (label_width - w)//2 + label_width, y - 20 - label_height - 10),
        (255, 255, 255),
        -1
        )

    frame = cv2.putText(
        frame,
        text,
        (x - (label_width - w)//2, y - 20),
        FONT,
        FONT_SCALE,
        (0, 0, 0),
        FONT_THICKNESS,
        cv2.LINE_AA
        )

    return frame

QR_DIR = 'qr_codes'
if QR_DIR not in os.listdir():
    os.mkdir(QR_DIR)

FONT_SCALE = .4 # font to print the transaction informations

qr = False # boolean to check if there is a QR Code within the video
already_paid = False # if qr code had the transaction Id, we could check if it was already paid with the post request

resol_array = (1024, 720) # camera resolution ~ lower is easier for the computer

# files with the OpenCv weights for the neural network that check faces and smiles
face_cascade = cv2.CascadeClassifier('cv2_cascades/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('cv2_cascades/haarcascade_smile.xml')

# start capturing video
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    # resize the frame so we can save computer energy
    frame = cv2.resize(frame, resol_array)

    # checks if the frame was correctly processed
    if ret:
        # creates a gray version of the frame to facilitate processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # looks for the QR code
        qr = pyzbar.decode(gray)

        # if the user shows the QR Code to the camera
        if qr:

            # creates QR code boundaries
            (x, y, w, h) = qr[0].rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # saves QR code data
            data = qr[0].data

            # checks if it has been paid
            if not already_paid:

                frame = write_contrast(frame, x, y, f'[SMILE TO PAY]', w = w, h = h)
                frame, smile = detect_smile(gray, frame)

                # pays after the user smiles
                if smile:
                    already_paid = True 

            # tells the user that the payment was successful over the QR code
            if already_paid:
                frame = write_contrast(frame, x, y, f'[SUCCESSFUL]', w = w, h = h)

        # user instructions
        elif not already_paid:
            frame = write_contrast(frame, resol_array[0]//2, 50, f'[SHOW YOUR QRCODE]')

        else:
            frame = write_contrast(frame, resol_array[0]//2, 50, f'[SUCCESSFULLY PAID]')

            # gets width and height of the largest text
            (label_width, label_height), baseline = cv2.getTextSize(show_text, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)

            # creates a big rectangle to create constrast for the transaction informations
            frame = cv2.rectangle(
                frame,
                (0, resol_array[1] - len(l_keys) * (label_height + 5)),
                (label_width + 5, resol_array[1]),
                (255, 255, 255),
                -1)

            # prints the transaction information for the user
            for i, key in enumerate(l_keys):
                frame = cv2.putText(
                    frame,
                    f"{key}: {d_req[key]}",
                    (5, resol_array[1] - i * (label_height + 5) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .4,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                    )

        # show the modified frame for the user
        cv2.imshow('frame', frame)

        # quits if the Q key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # quits the program if the image cannot be processed correctly
    else:
        break

# destroys OpenCV window
cv2.destroyAllWindows()
