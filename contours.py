# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:08:40 2019

@author: gasha
"""

import cv2 
import numpy as np

#720
#top= 90
#btm = 630
#lft = 200
#rgt = 1279

#1080
top= 100
btm = 980
lft = 100
rgt = 1820

## Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
    
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, img = vidObj.read()
        
        #decides which frames will be checked
        if count == 250:
            
            print(count)
            
            #crops img, speeds up runtime
            crop = img[top:btm,lft:rgt]
        
            try:
                temp1 = crop *1
                temp2 = crop*1
                temp3 = crop*1
            except:
                print("done")
        
            # process the image
            #  cape cod
#            gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) 
#            blur = cv2.GaussianBlur(gray,(21,21),0)
#            _,th = cv2.threshold(blur,150,255,cv2.THRESH_BINARY_INV)
#            kernel = np.ones((201,201),np.uint8)
#            opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
#            _, contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            # bob hall
            gray = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV) 
            blur = cv2.GaussianBlur(gray,(31,31),0)
            new= cv2.inRange(blur,(0,0,0),(20,255,255))
            kernel2 = np.ones((81,81),np.uint8)
            opening = cv2.morphologyEx(new, cv2.MORPH_OPEN, kernel2)
            fin =255 - opening
            kernel1 = np.ones((111,111),np.uint8)
            closing = cv2.morphologyEx(fin, cv2.MORPH_OPEN, kernel1)
            _, contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            c = max(contours, key = cv2.contourArea)
            #cv2.drawContours(img,[c],-1,(0, 255, 0),1)
            
            # get dimensions
            h,w,_=crop.shape
        
            shoreline = []
            for contour in c:
                if(contour[0][1] == 0 or contour[0][1] == h or contour[0][0] == w-1 or contour[0][0] == 0):
                    continue
                else:
                    shoreline.append(contour)
    
            for point in shoreline:
                crop[point[0][1],point[0][0]] = [255,0,255]
            
            # select area around contour
            f=0
            for k in range(h):
                for j in range(w):
                    if crop[k,j,2] == 255 and crop[k,j,0] == 255 and crop[k,j,1] == 0 and f==0:
                        p=j
                        f=1
                    if j == w-1:
                        if f==1:
                            for l in range(w):
                                if l < p+30:  #edit this line to change what part of the image is saved
                                    temp1[k,l] =[0,0,0]
                                if l > p-30:
                                    temp2[k,l] =[0,0,0]
                            l=0
                        if f==0:
                            for l in range(w):
                                temp1[k,l]= [0,0,0]
                j=0
                f=0
                    
            #presentation data generation
            cv2.imwrite('pres\\original%d.jpg' % count, temp3)
            cv2.imwrite('pres\\gray%d.jpg' % count, gray)
            cv2.imwrite('pres\\blur%d.jpg' % count, blur)
            cv2.imwrite('pres\\inversebinarythreshold%d.jpg' % count, fin)
            cv2.imwrite('pres\\opening%d.jpg' % count, closing)
            cv2.imwrite('pres\\contours%d.jpg' % count, crop)
            cv2.imwrite('pres\\final%d.jpg' % count, temp2)
            
            #standard data generation
            #edit the folder name to seperate data
#            cv2.imwrite('bhl1\\%d_land.jpg' % count, temp1)
#            cv2.imwrite('bhw1\\%d_water.jpg' % count, temp2)

        count += 1
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("bob_hall.mp4") 