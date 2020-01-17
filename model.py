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

    if len(blur.shape)==3:
    	gray = blur[:,:,0]
    else:
    	gray = blur

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

def use_lines(im):
	return rect_detect(im,line_detect(im))

def get_mask(im, sensitivity, center): # 36

	lower_hsv = np.array([center - sensitivity, 20, 20])
	upper_hsv = np.array([center + sensitivity, 255, 255])

	mask = cv2.inRange(im, lower_hsv, upper_hsv)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	def sharpen_image(image):
		image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
		image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
		return image_sharp

	sharpmask = sharpen_image(mask)
	return sharpmask.astype(int)

def plot_img(im):
	# plot the image and some data to help create good filters
	fig, ax = plt.subplots(6,5,figsize=(20,10))

	ax[0,0].set_title("Image")
	ax[0,0].imshow(im)

	ax[0,1].set_title("Vertical Average (top to bottom)")
	ax[0,1].plot(np.arange(im.shape[0]), np.mean(im[:,:,0], 1), c='red')
	ax[0,1].plot(np.arange(im.shape[0]), np.mean(im[:,:,1], 1), c='green')
	ax[0,1].plot(np.arange(im.shape[0]), np.mean(im[:,:,2], 1), c='blue')

	ax[0,2].set_title("Horizontal Average (left to right)")
	ax[0,2].plot(np.arange(im.shape[1]), np.mean(im[:,:,0], 0), c='red')
	ax[0,2].plot(np.arange(im.shape[1]), np.mean(im[:,:,1], 0), c='green')
	ax[0,2].plot(np.arange(im.shape[1]), np.mean(im[:,:,2], 0), c='blue')

	for i in range(5):
		for j in range(5):
			ax[i+1,j].imshow(get_mask(im, 30*i+80, 50+30*j))

	plt.show()
	return



im = cv2.cvtColor(cv2.imread('8.jpg'),cv2.COLOR_BGR2RGB)
#lines = line_detect(im)
#print(len(lines))

use_lines(im)

#ax[1].imshow(frangi(im[:,:,1],beta=1.5))
#ax[1].imshow(rect_detect(im,line_detect(im)), cmap='gray')
#ax[1].imshow(get_mask(im, 35))
plot_img(im)

