#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import os
import cv2
import cvlib as cv
import time 
import win32gui
import win32con
import win32api


# In[2]:


# download pre-trained model file (one-time download)
dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
model_path = get_file("gender_detection.model", dwnld_link,
                     cache_subdir="pre-trained", cache_dir=os.getcwd())

# load pre-trained model
model = load_model(model_path)


# In[3]:


from time import time

class Timer:
    def __init__(self,timer_period):
        
        self.timer_period = timer_period
        self.update_timer()
    
    def update_timer(self):
    
        self.last_time = time()
        self.timer_expires = self.last_time + self.timer_period
    
    def has_timer_expired(self):
        
        if time() > self.timer_expires:
            self.update_timer()
            return 1
        else:
            return 0
        
        
import multiprocessing , threading
def play_videoFile(name):
    vid = cv2.VideoCapture(name)
    
    
    while(vid.isOpened()):
        
        ret, frame = vid.read() 
        cv2.namedWindow(name[:-4],cv2.WND_PROP_FULLSCREEN)
#         cv2.moveWindow(name[:-4], 650,100)
        cv2.setWindowProperty(name[:-4], cv2.WND_PROP_FULLSCREEN, 
    cv2.WINDOW_FULLSCREEN)
#         cv2.resizeWindow(name[:-4],700,480)
        
        if name.startswith('default'):
            if ret:
                cv2.imshow(name[:-4], frame)
            else:
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            global stop_default
            if stop_default: 
                break
                
        if name.startswith('men'):
            if ret:
                cv2.imshow(name[:-4], frame)
            else:
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            global stop_men
            if stop_men: 
                break
                
        if name.startswith('women'):
            if ret:
                cv2.imshow(name[:-4], frame)
            else:
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            global stop_women
            if stop_women: 
                break
      
        
        if cv2.waitKey(1) & 0xFF == ord('b'):
            break   

    vid.release()
    cv2.destroyWindow(name[:-4])


# In[4]:


import time as tt
vote_dict = {'empty':0,'man':0,'woman':0}
t = Timer(2)
ad_output = ''

# open webcam
webcam = cv2.VideoCapture(0)
webcam.set(3, 320) #width
webcam.set(4, 216) #height


if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    
classes = ['woman','man']

stop_default = False
stop_men = True
stop_women = True

default_thread = threading.Thread(target=play_videoFile, args=('default.mp4',))
default_thread.start()

men_thread = threading.Thread(target=play_videoFile, args=('men.mp4',))

women_thread = threading.Thread(target=play_videoFile, args=('women.mp4',))



# th.start()
# loop through frames
while webcam.isOpened():
    
    cv2.namedWindow("Gender Detection")

    cv2.moveWindow("Gender Detection", 0,0)
    
    
#     cv2.resizeWindow("Gender Detection",640,480)
    hwnd = win32gui.FindWindow(None,'Gender Detection')
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,True)
#     win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)
#     win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0,0,0), 150, win32con.LWA_ALPHA)
    
    # read frame from webcam 
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    # apply face detection
    face, confidence = cv.detect_face(frame)
    
    #print(face)
    #print(confidence)
    if not face:
        
        vote_dict['empty']+=1
        if t.has_timer_expired():
            if (vote_dict['empty'] > vote_dict['man']) and (vote_dict['empty'] > vote_dict['woman'] ):
#                 print('No one Detected .... ')
                vote_dict = dict.fromkeys(vote_dict,0)
                
                if men_thread.is_alive():
                    stop_men = True
                if women_thread.is_alive():
                    stop_women = True
                
                if not default_thread.is_alive():
                    tt.sleep(0.5)
                    stop_default = False
                    default_thread = threading.Thread(target=play_videoFile, args=('default.mp4',))
                    default_thread.start()
                    
                    

    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0]
        #print(conf)
        #print(classes)
        
        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]
        ad_output = classes[idx]
#         print(ad_output)
        
        if 'woman' in ad_output:
            vote_dict['woman']+=1
            ad_output=''
            if t.has_timer_expired():
                if vote_dict['woman'] > vote_dict['man']  and vote_dict['woman'] > vote_dict['empty'] :
#                     print('W')
                    vote_dict = dict.fromkeys(vote_dict,0)
                    
                    if default_thread.is_alive():
                        stop_default = True
                    if men_thread.is_alive():
                        stop_men = True
                    if not women_thread.is_alive():
                        tt.sleep(0.5)
                        stop_women = False
                        women_thread = threading.Thread(target=play_videoFile, args=('women.mp4',))
                        women_thread.start()
                        
        if 'man' in ad_output:
            vote_dict['man']+=1
            ad_output=''
            if t.has_timer_expired():
                if vote_dict['man'] > vote_dict['woman']  and vote_dict['man'] > vote_dict['empty'] :
#                     print('M')
                    vote_dict = dict.fromkeys(vote_dict,4)
                    
                    
                    if default_thread.is_alive():
                        stop_default = True
                    if women_thread.is_alive():
                        stop_women = True
                    if not men_thread.is_alive():
                        tt.sleep(0.5)
                        stop_men = False
                        men_thread = threading.Thread(target=play_videoFile, args=('men.mp4',))
                        men_thread.start()
                                    
            
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    
    # display output    
    cv2.imshow("Gender Detection", frame)
     # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break
        
# release resources
webcam.release()
cv2.destroyWindow("Gender Detection")


# In[5]:


cv2. __version__


# In[6]:


import tensorflow as tf
tf. __version__


# In[ ]:




