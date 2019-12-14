# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 23:39:11 2019

@author: rossd
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import frangi

def line_detect(im):
    blur = cv2.blur(im,(5,5))
    gray = blur[:,:,0]
    high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 0.5*high_thresh, high_thresh)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold = 100, minLineLength = 400, maxLineGap = 50)        
    
    return lines
    
def rect_detect(im, lines):
    
    
    im2 = im.copy()
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(im2,(x1,y1),(x2,y2),(0,255,0),20)
    return im2

im = cv2.cvtColor(cv2.imread('1.jpg'),cv2.COLOR_BGR2RGB)
#lines = line_detect(im)
#print(len(lines))

fig, ax = plt.subplots(1,2,figsize=(14,8))

ax[0].imshow(im[:,:,1])
ax[1].imshow(frangi(im[:,:,1],beta1=1.5))
#ax[1].imshow(rect_detect(im,line_detect(im)), cmap='gray')