#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import dlib
import helper

show_flag = False
subject = ''
if sys.argv[1] == '-s' or sys.argv[1] == '--show':
    show_flag = True
    subject = sys.argv[2]
else:
    subject = sys.argv[1]

webcam = cv2.VideoCapture('videos/' + subject + '.avi')
if show_flag:
    cv2.namedWindow("Image")
num_frames = int(webcam.get(7))

f = open("data/" + subject + "_Feature.csv", "w")
f.write("Frame,Time (ms),Face Top Left (x),Face Top Left (y),Face Bottom Right (x),Face Bottom Left (y),Eye Left Center (x),Eye Left Center (y),Eye Right Center (x),Eye Right Center (y)")
file_data = []

for i in range(28,69):
    f.write(","+"Pt. "+str(i)+" (x)"+","+"Pt. "+str(i)+" (y)")
f.write("\n")

# Detector and predictor set up
modelFile = "detector/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "detector/deploy.prototxt.txt"
predictor = dlib.shape_predictor('predictor/shape_predictor_68_face_landmarks.dat')
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

for frame_num in range(num_frames):
    # We get a new frame from the webcam
    retval, frame = webcam.read()
    if retval:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        frameWidth  = webcam.get(3)  # float
        frameHeight = webcam.get(4) # float

        net.setInput(blob)
        detections = net.forward()
        x1, x2, y1, y2 = helper.find_face(detections, frameWidth, frameHeight)

        if show_flag:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 0, 0), 2)
        
        dlibRect = dlib.rectangle(x1, y1, x2, y2)
        shape = predictor(gray, dlibRect)
        shape = helper.shape_to_np(shape)
        
        # Find eye point extremes (corners)
        left_eye, right_eye = helper.face_extremes(shape)
        
        # Draw eye rectangles
        cv2.rectangle(frame,(left_eye[0]-10,left_eye[2]-10),((left_eye[1]+10,left_eye[3]+10)),(255,0,0),2)
        cv2.rectangle(frame,(right_eye[0]-10,right_eye[2]-10),((right_eye[1]+10,right_eye[3]+10)),(255,0,0),2)

        eye_left_center = ((left_eye[0]+left_eye[1])/2,(left_eye[2]+left_eye[3])/2)
        eye_right_center = ((right_eye[0]+right_eye[1])/2,(right_eye[2]+right_eye[3])/2)

        data_row = []
        data_row.append(str(webcam.get(1)))
        data_row.append(str(webcam.get(0)))
        data_row.append(str(x1))
        data_row.append(str(y1))
        data_row.append(str(x2))
        data_row.append(str(y2))
        data_row.append(str(eye_left_center[0]))
        data_row.append(str(eye_left_center[1]))
        data_row.append(str(eye_right_center[0]))
        data_row.append(str(eye_right_center[1]))
        for (x,y) in shape[27:68]:
            data_row.append(str(x))
            data_row.append(str(y))
            if show_flag:
                cv2.circle(frame,(int(x),int(y)),3,(255,0,0))
        file_data.append(data_row)

        if show_flag:
            cv2.imshow("Image", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("t"):
                print(webcam.get(1) / 1800)
        else:
            if frame_num % 9000 == 0:
                print(int(frame_num / 1800))

for row in file_data:
    for i in range(len(row)):
        if i != len(row) - 1:
            f.write(row[i]+',')
        else:
            f.write(row[i]+'\n')

f.close()
