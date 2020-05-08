#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from collections import OrderedDict

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def find_face(detections, frameWidth, frameHeight):
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
    return x1, x2, y1, y2


def face_extremes(shape):
    # min_x, max_x, min_y, max_y
    left_eye = [100000, 0, 100000, 0]
    right_eye = [100000, 0, 100000, 0]
    nose = [100000, 0, 100000, 0]
    mouth = [100000, 0, 100000, 0]
    
    for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
        if name == 'left_eye':
            for (x, y) in shape[i:j]:
                if x < left_eye[0]:
                    left_eye[0] = x
                if x > left_eye[1]:
                    left_eye[1] = x
                if y < left_eye[2]:
                    left_eye[2] = y
                if y > left_eye[3]:
                    left_eye[3] = y
        elif name == 'right_eye':
            for (x, y) in shape[i:j]:
                if x < right_eye[0]:
                    right_eye[0] = x
                if x > right_eye[1]:
                    right_eye[1] = x
                if y < right_eye[2]:
                    right_eye[2] = y
                if y > right_eye[3]:
                    right_eye[3] = y
        elif name == 'nose':
            for (x, y) in shape[i:j]:
                if x < nose[0]:
                    nose[0] = x
                if x > nose[1]:
                    nose[1] = x
                if y < nose[2]:
                    nose[2] = y
                if y > nose[3]:
                    nose[3] = y
                    
        elif name == 'mouth':
            for (x, y) in shape[i:j]:
                if x < mouth[0]:
                    mouth[0] = x
                if x > mouth[1]:
                    mouth[1] = x
                if y < mouth[2]:
                    mouth[2] = y
                if y > mouth[3]:
                    mouth[3] = y
                    
    return left_eye, right_eye, nose, mouth
