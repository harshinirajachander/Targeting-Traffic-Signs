#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 17:08:43 2018

@author: shehzeensh
"""

import cv2
import numpy as np
image_obj = cv2.imread('image.jpg')

gray = cv2.cvtColor(image_obj, cv2.COLOR_BGR2GRAY)

kernel = np.ones((4, 4), np.uint8)
dilation = cv2.dilate(gray, kernel, iterations=1)

blur = cv2.GaussianBlur(dilation, (5, 5), 0)


thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# Now finding Contours         ###################
_, contours, _ = cv2.findContours(
    thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
coordinates = []
for cnt in contours:
        # [point_x, point_y, width, height] = cv2.boundingRect(cnt)
    approx = cv2.approxPolyDP(
        cnt, 0.07 * cv2.arcLength(cnt, True), True)
    if len(approx) == 3:
        coordinates.append([cnt])
        cv2.drawContours(image_obj, [cnt], 0, (0, 0, 255), 3)

cv2.imwrite("result.png", image_obj)