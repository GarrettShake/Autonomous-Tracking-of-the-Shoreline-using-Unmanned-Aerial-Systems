# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:07:21 2019

@author: gasha
"""
import cv2
import numpy as np
import math
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import skimage
import time

def Build_Data_Set():
    data = pd.read_csv("v2.csv")
    X = data[ [
#               'Laplacian Mean',
               'Laplacian Standard Deviation',
#               'Canny Mean',
#               'Canny Standard Deviation',
               'Hue Mean',
#               'Blue Mean',
#               'Green Mean',
#               'Red Mean',
               'Shannons Entropy'
               ] ].values

    y = data["Class"].values

    return X,y

def train(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    X_train= MinMaxScaler().fit_transform(X_train)
    X_test= MinMaxScaler().fit_transform(X_test)

    clf.fit(X_train,y_train)

    print("Accuracy:", accuracy_score(y_test,clf.predict(X_test)))
    
    
def identify(img,X,clf,scaler):

    h,w,_ = img.shape
    
    laparr=[]
#    canarr=[]
    harr=[]
#    barr=[]
#    garr=[]
#    rarr=[]
    grayarr=[]
    
    #creates images for extraction
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_8U, ksize=15)
    hsv= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#    can=cv2.Canny(img,100,200)
        
    #adds wanted pixels to arrays
    for k in range(h):
        for j in range(w):
            laparr.append(lap[k,j])
#            canarr.append(can[k,j])
            harr.append(hsv[k,j,0])
#            barr.append(img[k,j,0])
#            garr.append(img[k,j,1])
#            rarr.append(img[k,j,2])
            grayarr.append(gray[k,j])
    
    #Extracts data
#    lapsum = sum(laparr)
#    lapmean = lapsum/len(laparr)
    lapdev = math.sqrt(np.var(laparr))
#    cansum = sum(canarr)
#    canmean = cansum/len(canarr)
#    candev = math.sqrt(np.var(canarr))
    hmean = sum(harr)/len(harr)
#    bmean = sum(barr)/len(barr)
#    gmean = sum(garr)/len(garr)
#    rmean = sum(rarr)/len(rarr)
    entropy = skimage.measure.shannon_entropy(grayarr)
    
    #puts data in Data Frame
    temp=pd.DataFrame({
#            'Laplacian Mean':[float(lapmean)],
            'Laplacian Standard Deviation':[float(lapdev)],
#            'Canny Mean':[float(canmean)],
#            'Canny Standard Deviation':[float(candev)],
            'Hue Mean':[float(hmean)],
#            'Blue Mean':[float(bmean)],
#            'Green Mean':[float(gmean)],
#            'Red Mean':[float(rmean)],
            'Shannons Entropy':[float(entropy)]
            }).values

    #normalization
    dat =scaler.transform(temp)
    
    #prediction
    return clf.predict(dat)[0]
    

clf = svm.SVC(kernel="sigmoid", C= 1.0,gamma='auto')  
X, y = Build_Data_Set()   
train(X,y)
scaler= MinMaxScaler()
scaler.fit(X)

# Path to video file 
vidObj = cv2.VideoCapture("cape_cod_edit.mp4") 
  
# Used as counter variable 
count = 0
  
# checks whether frames were extracted 
success = 1

# borders for cropping
top= 90
btm = 630
lft = 0
rgt = 1279

frames=15

t0 =time.time()  
while success: 
  
    # vidObj object calls read 
    # function extract frames 
    success, img = vidObj.read()
        
        
    if count % frames == 0:
        print("----------------------",count,"---------------------")
            
        crop = img[top:btm,lft:rgt]
        
        try:
            temp = crop *1
        except:
            print("done")
    
    
        # process the image
        gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(21,21),0)
        ret,th = cv2.threshold(blur,150,255,cv2.THRESH_BINARY_INV)
        kernel = np.ones((201,201),np.uint8)
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        _, contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        hc,wc,_=crop.shape
        for c in contours:
            x1,y1,w,h =cv2.boundingRect(c)
            x2 = x1+w
            y2 = y1+h
            tempimg=crop[y1:y2,x1:x2]
            cls = identify(tempimg,X,clf,scaler)
            print("Class:",cls)
            if cls == 0:
                cv2.drawContours(temp,[c],-1,(0, 0, 255),1)
#                for p in c:
#                    if(p[0][1] == 0 or p[0][1] == hc-1 or p[0][0] == 0 or p[0][0] == wc-1):
#                        temp[p[0][1],p[0][0]] = [255,0,0]
            elif cls == 1:
                cv2.drawContours(temp,[c],-1,(0, 255, 0),1)
        
        cv2.imwrite('fin2\\frame%d.jpg' % count, temp) 
    count+=1
    
t1=time.time()
t2=t1-t0                
print("total runtime:", t2)
print("average runtime:", t2/(count/frames))
                