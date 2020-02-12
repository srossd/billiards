# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 23:39:11 2019

@author: rossd
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import frangi

class LineSegment:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def pt1(self):
        return np.array([self.x1, self.y1])
    
    def pt2(self):
        return np.array([self.x2, self.y2])

    def mid(self):
        return 0.5*(self.pt1() + self.pt2())

    def dir(self):
        dpt = self.pt2()-self.pt1()
        return dpt/np.linalg.norm(dpt)

    def cos2(self, l):
        return (np.dot(self.dir(), l.dir()))**2

    def inter(self, l):
        A = np.array([[self.x2-self.x1, l.x1 - l.x2], [self.y2-self.y1, l.y1 - l.y2]])
        b = np.array([l.x1 - self.x1, l.y1 - self.y1])
        x = np.linalg.solve(A, b)
        return np.array([(1-x[0])*self.x1 + x[0]*self.x2, (1-x[0])*self.y1 + x[0]*self.y2])

    def extend(self):
        slope = (self.y2 - self.y1)/(self.x2-self.x1)
        return Line(slope, 1, -slope*self.x1 - self.y1)

    def __str__(self):
        return "(%.1f, %.1f) to (%.1f, %.1f)" % (self.x1, self.y1, self.x2, self.y2)

    def __repr__(self):
        return self.__str__()

class Line:
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
    
    def dist(self, pt):
        return np.abs((self.A*pt[0] + self.B*pt[1] + self.C)/np.sqrt(self.A*self.A + self.B*self.B + self.C*self.C))

def line_detect(im):
    blur = cv2.blur(im, (5, 5))

    if len(blur.shape) == 3:
        gray = blur[:, :, 0]
    else:
        gray = blur

    high_thresh, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 0.5*high_thresh, high_thresh)
    cv2lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=100, minLineLength=400, maxLineGap=50)

    lines = list(map(lambda x: LineSegment(*x[0]), cv2lines))

    return lines

def filter_lines(lines, cos2_thresh = .95, dist_thresh = 0.5):
    new_lines = []
    for l in lines:
        fresh = True
        for i in range(len(new_lines)):
            if new_lines[i].cos2(l) > cos2_thresh and new_lines[i].extend().dist(l.mid()) < dist_thresh:
                fresh = False
                break
        if fresh:
            new_lines.append(l)
    return new_lines

def draw_lines(im, lines):

    im2 = im.copy()
    for i, line in enumerate(lines):
        cv2.line(im2, (line.x1, line.y1), (line.x2, line.y2), (0, 255, 0), 20)
        cv2.putText(im2, str(i), (int(line.mid()[0]), int(line.mid()[1])), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 20)
    return im2

def draw_rect(im, r):
    im2 = im.copy()
    for i in range(len(r)):
        cv2.line(im2, (int(r[i][0]), int(r[i][1])), (int(r[(i+1) % len(r)][0]), int(r[(i+1) % len(r)][1])), (0, 255, 0), 20)
        cv2.putText(im2, str(i), (int(r[i][0]), int(r[i][1])), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 20)
    return im2

def find_rect(l1, l2, l3, l4):
    xs = [(x, l1.cos2(x)) for x in (l2, l3, l4)]
    xs.sort(key = lambda x: x[1])
    pairs = [(l1, xs[0][0]), (xs[0][0], xs[2][0]), (xs[2][0], xs[1][0]), (xs[1][0], l1)]

    rect_vertices = np.float32(list(map(lambda x: x[0].inter(x[1]), pairs[:4])))
    return rect_vertices

def extract_table(im, r):
    pts = np.float32([[0,2000],[1000,2000],[1000, 0],[0,0]])

    M = cv2.getPerspectiveTransform(r,pts)
    return cv2.warpPerspective(im,M,(1000, 2000))

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
	'''
	hull = contour(Normalize(dots))

	# draw contours and hull points
	for i in range(len(hull)):
	    color = (255, 0, 0) # blue - color for convex hull
	    # draw ith convex hull object
	    cv2.drawContours(im, hull, i, color, 1, 8)

	ax[0,3].imshow(imContour)
	'''	
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


def get_mask(im, sensitivity, center):  # 36

    lower_hsv = np.array([center - sensitivity, 20, 20])
    upper_hsv = np.array([center + sensitivity, 255, 255])

    mask = cv2.inRange(im, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def sharpen_image(image):
        image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
        image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
        return image_sharp

    sharpmask = sharpen_image(mask)
    return sharpmask.astype(int)


def plot_img(im):
    # plot the image and some data to help create good filters
    fig, ax = plt.subplots(6, 5, figsize=(20, 10))

    ax[0, 0].set_title("Image")
    ax[0, 0].imshow(im)

    ax[0, 1].set_title("Vertical Average (top to bottom)")
    ax[0, 1].plot(np.arange(im.shape[0]), np.mean(im[:, :, 0], 1), c='red')
    ax[0, 1].plot(np.arange(im.shape[0]), np.mean(im[:, :, 1], 1), c='green')
    ax[0, 1].plot(np.arange(im.shape[0]), np.mean(im[:, :, 2], 1), c='blue')

    ax[0, 2].set_title("Horizontal Average (left to right)")
    ax[0, 2].plot(np.arange(im.shape[1]), np.mean(im[:, :, 0], 0), c='red')
    ax[0, 2].plot(np.arange(im.shape[1]), np.mean(im[:, :, 1], 0), c='green')
    ax[0, 2].plot(np.arange(im.shape[1]), np.mean(im[:, :, 2], 0), c='blue')

    for i in range(5):
        for j in range(5):
            ax[i+1, j].imshow(get_mask(im, 30*i+80, 50+30*j))

    plt.show()
    return


im = cv2.cvtColor(cv2.imread('images/1.jpg'), cv2.COLOR_BGR2RGB)
lines = line_detect(im)
flines = filter_lines(lines)

r = find_rect(flines[1], flines[2], flines[3], flines[4])

fig, ax = plt.subplots(1, 3, figsize=(20, 7))
ax[0].imshow(draw_lines(im, flines), cmap='gray')

# # ax[1].imshow(frangi(im[:,:,1],beta=1.5))
ax[1].imshow(draw_rect(im, r), cmap='gray')
ax[2].imshow(extract_table(im, r), cmap='gray')
plt.show()
# #ax[1].imshow(get_mask(im, 35))
# plot_img(im)

im = cv2.cvtColor(cv2.imread('images/2.jpg'),cv2.COLOR_BGR2RGB)
plot_img(im)