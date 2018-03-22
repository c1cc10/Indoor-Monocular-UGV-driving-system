# -*- coding: utf-8 -*-
import numpy as np
import cv2
import time
from numpy import *
from utility import *

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

    
class RoadDetection:
    
    def __init__(self, image_source):
        self.cap = image_source
        self.cap.set(3,640)
        self.cap.set(4,480)

        self.StepSize = 8

        self.filters = build_filters()
        ret, self.frame = cap.read()
        self.filters = []
        self.right_lanes = []
        self.left_lanes = []
        self.lines = []
        self.imagewidth = self.frame.shape[1] - 1
        self.imageheight = self.frame.shape[0] - 1
        self.middle_width = self.imagewidth/2
        self.middle_height = self.imageheight/2
        self.one_third_height = self.imageheight/3
        self.two_thirds_height = self.one_third_height*2 
        self.p1 = array([0, self.two_thirds_height])
        self.p2 = array([self.imagewidth, self.two_thirds_height])
        self.p4 = array([self.imagewidth, self.imageheight])
        self.p5 = array([0, self.imageheight])
        self.delta = 0
        self.van_x = 0
        self.vanishing_point = 0, 0

    def find_obstacles_position(self, obstacles, myframe):
        for hurdle in obstacles:
            colore = (0,255,255)
            for bisX in get_line(self.vanishing_point, (self.middle_width, self.imageheight)):
                if hurdle.endPoint < bisX: # obstacle is on the left
                    colore = (0,0,255)
                    self.delta += 1 # turn right
                else: # obstacle on the right
                    if hurdle.startPoint < bisX: #hurdle in middle path
                        pass #print "middle path hurdle"
                    self.delta -= 1 #turn left 
            cv2.circle(frame, hurdle.endPoint, 6, colore)        
            #cv2.circle(frame, pippo.startPoint, 5, (0,255,255))
            #cv2.circle(frame, pippo.endPoint, 6, (0,0,255))
        if len(obstacles) == 0:
            self.delta = 0

    def draw_line (self, blk_bgd):
        #print right
        if len(self.right_lanes) > 0:
            vx, vy, cx, cy = cv2.fitLine(np.array(self.right_lanes), cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
            cv2.line(blk_bgd,(int(cx-vx*self.imagewidth), int(cy-vy*self.imagewidth)), (int(cx+vx*self.imagewidth), int(cy+vy*self.imagewidth)), [0, 255, 0], 2)
        if len(self.left_lanes) > 0:
            vx, vy, cx, cy = cv2.fitLine(np.array(self.left_lanes), cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
            cv2.line(blk_bgd,(int(cx-vx*self.imagewidth), int(cy-vy*self.imagewidth)), (int(cx+vx*self.imagewidth), int(cy+vy*self.imagewidth)), [0, 0, 255], 2)
        return blk_bgd

    def draw_lane_lines_probabilistic(self):
        black_bgd =  np.zeros((self.frame.shape[0], self.frame.shape[1], 3), dtype=np.uint8)
        if self.lines is None:
            return black_bgd

        for line in self.lines:
            x1,y1,x2,y2 = line[0]
            if x2 == x1: #remove vertical lines (undefined result or dividing by zero)
                continue
            slope = (float(y2) - y1) / (x2 - x1)
            if slope == 0: # vertical line
                continue
            if math.fabs(slope) < 0.7: #or math.fabs(slope) > 1: # <-- Only consider extreme slope
                continue
            if slope > 0 :
                self.right_lanes.append(np.array([x1,y1+self.one_third_height]))
                self.right_lanes.append(np.array([x2,y2+self.one_third_height]))
            else:
                self.left_lanes.append(np.array([x1,y1+self.one_third_height]))
                self.left_lanes.append(np.array([x2,y2+self.one_third_height]))
        return self.draw_line(black_bgd)

    def get_lanes(self,img):
        self.lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=15, lines=np.array([]), minLineLength=10, maxLineGap=30)
        line_img = self.draw_lane_lines_probabilistic()
        #lines = cv2.HoughLines(img, 2, np.pi/180, 20)
        #line_img = draw_lane_lines_deterministic(lines, frame)
        return cv2.addWeighted(line_img, 0.8, self.frame, 1.0, 0.0)

    def run(self):
        while(True):
            self.van_x = self.middle_width + self.delta
            if self.van_x > self.imagewidth:
                self.van_x = self.imagewidth
            if self.van_x < 0:
                self.van_x = 0
            obstacles = []
            newObstacle = Obstacle((0,0),(0,0))
            #obstacles.append(newObstacle)
            EdgeArray = []
            #time.sleep(0.1)#let image settle
            ret, self.frame = self.cap.read()
            roi = self.frame[self.one_third_height:self.imageheight, 0:self.imagewidth]
            #res2 = process(roi[:,:,2], filters)
            self.vanishing_point = (self.van_x,self.middle_height)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # or gray = cv2.bilateralFilter(gray,9,30,30)
            gray = cv2.GaussianBlur(gray,(15,15),0)
            img = cv2.Canny(gray, 50, 100) 
            imgw = img.shape[1] - 1
            imgh = img.shape[0] - 1
            
            # Disegniamo il vanishing point
            self.frame = self.get_lanes(img)
            cv2.circle(self.frame, self.vanishing_point, 2, (255,34,233))
            # Disegniamo la road extraction area    
            intersect_left = seg_intersect( self.p1, self.p2, self.vanishing_point, self.p4)
            intersect_right = seg_intersect( self.p1, self.p2, self.vanishing_point, self.p5)
            road_extraction_area = np.array([[0,self.imageheight],intersect_right, intersect_left,[self.imagewidth,self.imageheight]], dtype=np.int32)    
            cv2.drawContours(self.frame, [road_extraction_area], 0,(255,255,255),2)
            for j in range (0,imgw,self.StepSize):    
                for i in range(imgh-3,0,-1):    
                    if img.item(i,j) == 255:       
                        if not cv2.pointPolygonTest(road_extraction_area, (j,i+self.one_third_height), False) >= 0:
                            break
                        if not len(EdgeArray) == 0 and j - EdgeArray[len(EdgeArray)-1][0] == self.StepSize:
                            obstacles[len(obstacles)-1].endPoint = (j,i+self.one_third_height)
                        else:
                            newObstacle = Obstacle((j,i+self.one_third_height), (0,0))
                            obstacles.append(newObstacle) 
                        EdgeArray.append((j,i+self.one_third_height))        #if it is, add x,y coordinates to ObstacleArray
                        break                          #if white pixel is found, skip rest of pixels in column
            for pxl in range (len(EdgeArray)-1):      #draw lines between points in ObstacleArray       
                if EdgeArray[pxl+1][0] - EdgeArray[pxl][0] == self.StepSize: # allora sono un unico oggetto
                    cv2.line(self.frame, EdgeArray[pxl], EdgeArray[pxl+1],(255,0,0),1)
                    #cv2.line(frame, (EdgeArray[pxl][0], imageheight), EdgeArray[pxl],(0,255,0),1)
            #delta = find_obstacles_position(obstacles, frame, vanishing_point, delta)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            self.show()
            
    def show(self):    
        # Display the resulting frame
        cv2.imshow('frame',self.frame)
        #cv2.imshow('gabor wavelets',res2)
        #cv2.imshow('canny', img)

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows

#cap = cv2.VideoCapture("bici.m4v")
cap = cv2.VideoCapture(1)
pippo = RoadDetection(cap)
pippo.run()
