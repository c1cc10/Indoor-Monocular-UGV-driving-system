# -*- coding: utf-8 -*-
import numpy as np
import cv2
import time
from numpy import *

side = ['unknown','left', 'center', 'right']
highness = ['unknown', 'high', 'middle', 'low']

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 10):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

class Obstacle:
    startPoint = None
    endPoint = None #array([0,0])
    position = None #side[0], highness[0]
    
    def __init__(self, start, end):
        self.startPoint = start
        self.endPoint = end
        #self.position = array(side['unknown'], highness['unknown'])
    def changeHighness(self, newhigh):
        self.position[1] = highness[newhigh]
    def changeSide(self, newSide):
        self.position[0] = side[newside]

    
cap = cv2.VideoCapture("bici.m4v")
cap.set(3,640)
cap.set(4,480)

StepSize = 8
EdgeArray = []

filters = build_filters()
ret, frame = cap.read()

imagewidth = frame.shape[1] - 1
imageheight = frame.shape[0] - 1
middle_width = imagewidth/2
middle_height = imageheight/2
one_third_height = imageheight/3
two_thirds_height = one_third_height*2 
p1 = array([0, two_thirds_height])
p2 = array([imagewidth, two_thirds_height])
p4 = array([imagewidth, imageheight])
p5 = array([0, imageheight])
delta = 0

#Bresenham's Line Algorithm
def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
 
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

def find_obstacles_position(obstacles, myframe, vanishing_point, mydelta):
    for hurdle in obstacles:
        colore = (0,255,255)
        for bisX in get_line(vanishing_point, (middle_width, imageheight)):
            if hurdle.endPoint < bisX: # obstacle is on the left
                colore = (0,0,255)
                mydelta += 1 # turn right
            else: # obstacle on the right
                if hurdle.startPoint < bisX: #hurdle in middle path
                    pass #print "middle path hurdle"
                mydelta -= 1 #turn left 
        cv2.circle(frame, hurdle.endPoint, 6, colore)        
        #cv2.circle(frame, pippo.startPoint, 5, (0,255,255))
        #cv2.circle(frame, pippo.endPoint, 6, (0,0,255))
    if len(obstacles) == 0:
        mydelta = 0
    return mydelta

# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#
def perp( a ) :
    b = empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = dot( dap, db)
    num = dot( dap, dp )
    return (num / denom.astype(float))*db + b1

while(True):
    van_x = middle_width + delta
    if van_x > imagewidth:
        van_x = imagewidth
    if van_x < 0:
        van_x = 0
    obstacles = []
    newObstacle = Obstacle((0,0),(0,0))
    #obstacles.append(newObstacle)
    EdgeArray = []
    #time.sleep(0.1)#let image settle
    # Capture frame-by-frame
    ret, frame = cap.read()
    roi = frame[one_third_height:imageheight, 0:imagewidth]
    res2 = process(roi[:,:,2], filters)
    vanishing_point = (van_x,middle_height)
    # Our operations on the frame come here
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #gray = cv2.bilateralFilter(gray,9,30,30)
    gray = cv2.GaussianBlur(res2,(15,15),0)
    img = cv2.Canny(gray, 50, 100)             #edge detection
    imgw = img.shape[1] - 1
    imgh = img.shape[0] - 1
    
    # Disegniamo il vanishing point
    # LAne recognition code from here ...
    #lines = cv2.HoughLinesP(img, 2, np.pi/180, 80, np.array([]), minLineLength=30, maxLineGap=10)
    #lines = cv2.HoughLines(img, 2, np.pi/180, 20)
    #line_img =  np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    #if not lines is None:
    #    for line in lines:
    #        for x1,y1,x2,y2 in line:
    #            cv2.line(frame, (x1, y1+((imageheight/3)*2)), (x2, y2+((imageheight/3)*2)), [0, 255, 0], 2)
    #    puppa = cv2.addWeighted(line_img, 0.8, frame, 1.0, 0.0)
    # ... to here
    cv2.circle(frame, vanishing_point, 2, (255,34,233))
    # Disegniamo la road extraction area    
    intersect_left = seg_intersect( p1,p2, vanishing_point,p4)
    intersect_right = seg_intersect( p1,p2, vanishing_point,p5)
    road_extraction_area = np.array([[0,imageheight],intersect_right, intersect_left,[imagewidth,imageheight]], dtype=np.int32)    
    cv2.drawContours(frame, [road_extraction_area], 0,(255,255,255),2)
    for j in range (0,imgw,StepSize):    #for the width of image array
        for i in range(imgh-3,0,-1):    #step through every pixel in height of array from bottom to top. Ignore first couple of pixels as may trigger due to undistort
            if img.item(i,j) == 255:       #check to see if the pixel is white which indicates an edge has been found
                if not cv2.pointPolygonTest(road_extraction_area, (j,i+((imageheight/3)*2)), False) >= 0:
                    break
                if not len(EdgeArray) == 0 and j - EdgeArray[len(EdgeArray)-1][0] == StepSize:
                    obstacles[len(obstacles)-1].endPoint = (j,i+((imageheight/3)*2))
                    #print "endPoint: %d, %d" % (j,i+((imageheight/3)*2))
                else:
                    #print "startPoint: %d, %d" % (j,i+((imageheight/3)*2))
                    newObstacle = Obstacle((j,i+((imageheight/3)*2)), (0,0))
                    obstacles.append(newObstacle) 
                EdgeArray.append((j,i+((imageheight/3)*2)))        #if it is, add x,y coordinates to ObstacleArray
                break                          #if white pixel is found, skip rest of pixels in column
    for pxl in range (len(EdgeArray)-1):      #draw lines between points in ObstacleArray       
        if EdgeArray[pxl+1][0] - EdgeArray[pxl][0] == StepSize: # allora sono un unico oggetto
            cv2.line(frame, EdgeArray[pxl], EdgeArray[pxl+1],(255,0,0),1)
            #cv2.line(frame, (EdgeArray[pxl][0], imageheight), EdgeArray[pxl],(0,255,0),1)
    #cv2.line(frame,vanishing_point, (middle_width, imageheight), (255,255,255))
    delta = find_obstacles_position(obstacles, frame, vanishing_point, delta)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('gabor wavelets',res2)
    cv2.imshow('canny', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows
