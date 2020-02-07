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
'''
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
'''

def contour(im):
		
	gray = np.array(im, dtype = np.uint8)
	blur = cv2.blur(gray, (3, 3)) # blur the image
	ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
	im2, contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	#print(contours)
	contours = contours[0]
	# create hull array for convex hull points
	hull = []

	# calculate points for each contour
	for i in range(len(contours)):
	    # creating convex hull object for each contour
	    hull.append(c2.convexHull(contours[i]))
	
	#hull = cv2.convexHull(contours)

	print(hull)
	return hull


def plot_img(im):
	# plot the image and some data to help create good filters
	fig, ax = plt.subplots(2,4,figsize=(20,20))
	'''
	ax[0,0].set_title("Image")
	ax[0,0].imshow(im)
	'''
	'''
	ax[0,1].set_title("Vertical Average (top to bottom)")
	ax[0,1].plot(np.arange(im.shape[0]), np.mean(im[:,:,0], 1), c='red')
	ax[0,1].plot(np.arange(im.shape[0]), np.mean(im[:,:,1], 1), c='green')
	ax[0,1].plot(np.arange(im.shape[0]), np.mean(im[:,:,2], 1), c='blue')

	ax[0,2].set_title("Horizontal Average (left to right)")
	ax[0,2].plot(np.arange(im.shape[1]), np.mean(im[:,:,0], 0), c='red')
	ax[0,2].plot(np.arange(im.shape[1]), np.mean(im[:,:,1], 0), c='green')
	ax[0,2].plot(np.arange(im.shape[1]), np.mean(im[:,:,2], 0), c='blue')
	'''
	'''
	for i in range(5):
		for j in range(5):
			ax[i+1,j].imshow(get_mask(im, 30*i+80, 50+30*j))
	'''
	ax[1,0].imshow(im)
	dots, harrisImg = Harris_Corner(im)

	#cv2.imshow('dst',img)
	ax[0,0].imshow(harrisImg)
	ax[0,1].imshow(Normalize(dots), cmap='Greys')

	ax[0,2].imshow(use_lines(im))
	hull = contour(Normalize(dots))

	# draw contours and hull points
	for i in range(len(hull)):
	    color = (255, 0, 0) # blue - color for convex hull
	    # draw ith convex hull object
	    cv2.drawContours(im, hull, i, color, 1, 8)

	ax[0,3].imshow(imContour)
	
	plt.show()
	return

def Normalize(img):
	min = np.amin(img)
	max = np.amax(img)
	return 255.0*(img-min)/(max-min)

def Harris_Corner(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,50,3,0.04)
	
	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)

	# Threshold for an optimal value, it may vary depending on the image.
	#hImg = np.zeros((img.shape))
	hImg = np.copy(img)
	hImg[dst>0.01*dst.max()]=[0,0,255]

	dots = np.zeros((img.shape[0], img.shape[1]))
	dots[dst>0.01*dst.max()] = 1

	return dots, hImg



im = cv2.cvtColor(cv2.imread('2.jpg'),cv2.COLOR_BGR2RGB)






plot_img(im)







