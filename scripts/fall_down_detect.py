import os
import sys
import argparse
import ast
import cv2
import time
import torch
from vidgear.gears import CamGear
import numpy as np

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import ffmpeg
from math import acos, degrees


class FallDown:

    def __init__(self):
    
        self.frame_count =0
        self.situp_count =0
        self.down_count =0
        self.falldown_flag = False
        self.frame_delay =0
        self.now_x = 0
        self.now_y = 0


    def check_fall_down(self, image, points, skeleton , video_framerate):
        self.frame_count += 1
        height, width, channels = image.shape
        
        #0: "nose", 15: "left_ankle",16: "right_ankle"
        #13: "left_knee", 14: "right_knee",
        ave_ankle_x = 0
        ave_ankle_y = 0
        
        if points[15][2] > 0.5 and  points[16][2]:
            ave_ankle_y = (points[15][0] + points[16][0]) /2
            ave_ankle_x = (points[15][1] + points[16][1]) /2
        elif points[15][2] > 0.5 :
            ave_ankle_y = points[15][0]
            ave_ankle_x = points[15][1]
        elif points[16][2] > 0.5 :
            ave_ankle_y = points[16][0]
            ave_ankle_x = points[16][1]
        elif points[15][2] > 0.5 :
            ave_ankle_y = points[13][0]
            ave_ankle_x = points[13][1]
        elif points[14][2] > 0.5 :
            ave_ankle_y = points[14][0]
            ave_ankle_x = points[14][1]
        else:
            ave_ankle_y = points[12][0]
            ave_ankle_x = points[12][1]
            
        if points[0][2] > 0.5 :
            ave_face_y = points[0][0]
            ave_face_x = points[0][1]
        else:
            ave_face_y = points[1][0]
            ave_face_x = points[1][1]
        
            
        dif_x = abs(points[0][1] - ave_ankle_x)
        if dif_x == 0:
            dif_x = 0.1
        
        dif_y = abs(points[0][0] - ave_ankle_y)
        if dif_y == 0:
            dif_y = 0.1
        
            
        degree = (dif_y - dif_x) / dif_x
        #print("\ndegree : ", degree, " dif_x:", dif_x, " dif_y:",dif_y)
        
        
        if points[16][2] > 0.5:
            self.now_x = int(points[16][1])
            self.now_y = int(points[16][0] + 30)
        
        
        if degree >= 0 :
            self.situp_count += 1
            self.falldown_flag = False
            #if self.frame_delay >0:
                #self.frame_delay -= 1
                #cv2.putText(image,"Fall down!!!", ( self.now_x,self.now_y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2 )
                
            
        else:
            self.down_count += 1
            #self.frame_delay = 5
            #print("Fall down!!!")
            cv2.putText(image,"Fall down!!!", (self.now_x,self.now_y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2 )
            
            '''
            if self.down_count > video_framerate and self.situp_count > video_framerate:
                self.falldown_flag = True
                self.situp_count = 0
                
            if self.falldown_flag == True:
                print("Fall down!!!")
                cv2.putText(image,"Fall down!!!", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0) )
            '''
    
    
        return image