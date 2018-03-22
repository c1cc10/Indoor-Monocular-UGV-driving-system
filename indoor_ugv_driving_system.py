# -*- coding: utf-8 -*-
import numpy as np
import cv2
import time
from numpy import *
from utility import *

cap = cv2.VideoCapture("bici.m4v")
#cap = cv2.VideoCapture(1)

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

def draw_line (right, left, frame):
    #print right
    if len(right) > 0:
        vx, vy, cx, cy = cv2.fitLine(np.array(right), cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
        cv2.line(frame,(int(cx-vx*imagewidth), int(cy-vy*imagewidth)), (int(cx+vx*imagewidth), int(cy+vy*imagewidth)), [0, 255, 0], 2)
    if len(left) > 0:
        vx, vy, cx, cy = cv2.fitLine(np.array(left), cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
        cv2.line(frame,(int(cx-vx*imagewidth), int(cy-vy*imagewidth)), (int(cx+vx*imagewidth), int(cy+vy*imagewidth)), [0, 0, 255], 2)
    #    cv2.line(frame, (lx1, ly1), (lx2, ly2), [0, 0, 255], 2)
    #vx, vy, cx, cy = cv2.fitLine(np.float32(points), func, 0, 0.01, 0.01)
    #cv2.line(frame, (int(cx-vx*w), int(cy-vy*w)), (int(cx+vx*w), int(cy+vy*w)), (0, 0, 255))
    #frame = draw_lines(x1,y1,x2,y2)
    return frame

def draw_lane_lines_probabilistic(lines, img):
    frame =  np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is None:
        return frame
    right = []
    left = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        if x2 == x1: #remove vertical lines (undefined result or dividing by zero)
            continue
        slope = (float(y2) - y1) / (x2 - x1)
        if slope == 0: # vertical line
            continue
#        print slope
        if math.fabs(slope) < 0.7: #or math.fabs(slope) > 1: # <-- Only consider extreme slope
            continue
        if slope > 0 :
            right.append(np.array([x1,y1+one_third_height]))
            right.append(np.array([x2,y2+one_third_height]))
        else:
            left.append(np.array([x1,y1+one_third_height]))
            left.append(np.array([x2,y2+one_third_height]))
    frame = draw_line(right, left ,frame)
    return frame

def get_lanes(img, frame):
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=15, lines=np.array([]), minLineLength=10, maxLineGap=30)
    line_img = draw_lane_lines_probabilistic(lines, frame)
    #lines = cv2.HoughLines(img, 2, np.pi/180, 20)
    #line_img = draw_lane_lines_deterministic(lines, frame)
    return cv2.addWeighted(line_img, 0.8, frame, 1.0, 0.0)

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
    #res2 = process(roi[:,:,2], filters)
    vanishing_point = (van_x,middle_height)
    # Our operations on the frame come here
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #gray = cv2.bilateralFilter(gray,9,30,30)
    gray = cv2.GaussianBlur(gray,(15,15),0)
    img = cv2.Canny(gray, 50, 100)             #edge detection
    imgw = img.shape[1] - 1
    imgh = img.shape[0] - 1
    
    # Disegniamo il vanishing point
    # LAne recognition code from here ...
    frame = get_lanes(img, frame)
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
                if not cv2.pointPolygonTest(road_extraction_area, (j,i+one_third_height), False) >= 0:
                    break
                if not len(EdgeArray) == 0 and j - EdgeArray[len(EdgeArray)-1][0] == StepSize:
                    obstacles[len(obstacles)-1].endPoint = (j,i+one_third_height)
                else:
                    newObstacle = Obstacle((j,i+one_third_height), (0,0))
                    obstacles.append(newObstacle) 
                EdgeArray.append((j,i+one_third_height))        #if it is, add x,y coordinates to ObstacleArray
                break                          #if white pixel is found, skip rest of pixels in column
    for pxl in range (len(EdgeArray)-1):      #draw lines between points in ObstacleArray       
        if EdgeArray[pxl+1][0] - EdgeArray[pxl][0] == StepSize: # allora sono un unico oggetto
            cv2.line(frame, EdgeArray[pxl], EdgeArray[pxl+1],(255,0,0),1)
            #cv2.line(frame, (EdgeArray[pxl][0], imageheight), EdgeArray[pxl],(0,255,0),1)
    #delta = find_obstacles_position(obstacles, frame, vanishing_point, delta)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    #cv2.imshow('gabor wavelets',res2)
    #cv2.imshow('canny', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows
