#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import dlib
import numpy as np
import helper

subject = sys.argv[1]
webcam = cv2.VideoCapture('videos/' + subject + '.avi')

f = open("data/" + subject + "_Feature.csv", "w")
f.write("Frame,Face Top Left (x),Face Top Left (y),Face Bottom Right (x),Face Bottom Left (y),Eye Left Center (x),Eye Left Center (y),Eye Right Center (x),Eye Right Center (y),Nose Center (x),Nose Center (y),Mouth Center (x),Mouth Center (y)\n")

# Detector and predictor set up
modelFile = "detector/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "detector/deploy.prototxt.txt"
predictor = dlib.shape_predictor('predictor/shape_predictor_68_face_landmarks.dat')
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    frameWidth  = webcam.get(3)  # float
    frameHeight = webcam.get(4) # float

    net.setInput(blob)
    detections = net.forward()
    x1, x2, y1, y2 = helper.find_face(detections, frameWidth, frameHeight)

    cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 0, 0), 2)
    
    dlibRect = dlib.rectangle(x1, y1, x2, y2)
    shape = predictor(gray, dlibRect)
    shape = helper.shape_to_np(shape)
    
    # Find eye point extremes (corners)
    left_eye, right_eye, nose, mouth = helper.face_extremes(shape)
    
    # Draw eye rectangles
    cv2.rectangle(frame,(left_eye[0]-10,left_eye[2]-10),((left_eye[1]+10,left_eye[3]+10)),(255,0,0),2)
    cv2.rectangle(frame,(right_eye[0]-10,right_eye[2]-10),((right_eye[1]+10,right_eye[3]+10)),(255,0,0),2)

    # Draw nose and mouth
    cv2.circle(frame,(int((nose[0]+nose[1])/2),int((nose[2]+nose[3])/2)),6,(255,0,0))
    cv2.circle(frame,(int((mouth[0]+mouth[1])/2),int((mouth[2]+mouth[3])/2)),6,(255,0,0)) 
    
    eye_left_center = ((left_eye[0]+left_eye[1])/2,(left_eye[2]+left_eye[3])/2)
    eye_right_center = ((right_eye[0]+right_eye[1])/2,(right_eye[2]+right_eye[3])/2)
    nose_center = ((nose[0]+nose[1])/2,(nose[2]+nose[3])/2)
    mouth_center = ((mouth[0]+mouth[1])/2,(mouth[2]+mouth[3])/2)
    f.write(str(webcam.get(1))+","+str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+str(eye_left_center[0])+","+str(eye_left_center[1])+","+str(eye_right_center[0])+","+str(eye_right_center[1])+","+str(nose_center[0])+","+str(nose_center[1])+","+str(mouth_center[0])+","+str(mouth_center[1])+"\n")
    cv2.imshow("Image", frame)
    
    if cv2.waitKey(1) == 27:
        break
    if cv2.waitKey(1) == 116:
        print(webcam.get(1))

f.close()