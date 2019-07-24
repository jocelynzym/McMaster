# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:37:37 2019

@author: xiong
"""
import numpy as np
import cv2

class particle:
    MAX_F= 7 #particle class save the positon of this particle in the previous frames. MAX_F is how many frames saved (include the current one).
    MAX_fl_img= 6 #particle class save the flourescent images of this particle in the previous frames. MAX_fl_img is how many frames saved (include the current one).
    FL_SIZE=100
    tem=np.zeros((FL_SIZE,FL_SIZE)) 
    mask=np.zeros((FL_SIZE,FL_SIZE)) 
    bbox_pred = [0,0,0,0]
    
    def __init__(self, bbox,img):
      self.bbox = bbox
      self.img = img
      self.position_x = [-100,-100,-100]
      self.position_y = [-100,-100,-100]
      self.frame= [0, 0, 0]
      self.frame_fl= [0, 0]
      self.speed = [0, 0]
      self.appear = 3
      self.img_fl = np.zeros((particle.FL_SIZE,particle.FL_SIZE)) 
      self.img_fl= np.append(self.img_fl, self.img_fl, axis = 0)
      self.position_prediction = [-1,-1]
      self.kalman_ok= 0
      self.kalman_init_num= 2
      #for tracking method: find matching in a square
      self.template=0
      self.img_tracking=0
      self.box_ps=0
      self.tracking_l=1
      self.tracking_ok = 1
      #a bbox obtained from hog which is closed to tracker most
      self.hog_bbox = (0,0,0,0);
      #distance between hog position and tracking position
      self.distance_t2h = 500
      #distance between kalman position and tracking position
      self.distance_k2t = 500
      #distance between kalman position and hog position
      self.distance_k2h = 500
      #save index of hog detector
      self.index_2h = -1
      self.case=0
      self.type = 'unknown'
      self.PN = 0
      #0 is unknow 
      #1 is beam
      #2 is algae
      
      
    def assign_PN (self,num):
        if self.PN == 0:
            self.PN = num
            return 1
        else :
            return 0
        
        
    def tracker_create(self):
        self.tracker = cv2.TrackerCSRT_create()
#if tracker_type == 'BOOSTING':
#    tracker = cv2.TrackerBoosting_create()
#if tracker_type == 'MIL':
#    tracker = cv2.TrackerMIL_create()
#if tracker_type == 'KCF':
#    tracker = cv2.TrackerKCF_create()
#if tracker_type == 'TLD':
#    tracker = cv2.TrackerTLD_create()
#if tracker_type == 'MEDIANFLOW':
#    tracker = cv2.TrackerMedianFlow_create()
#if tracker_type == 'GOTURN':
#    tracker = cv2.TrackerGOTURN_create()
#if tracker_type == 'MOSSE':
#    tracker = cv2.TrackerMOSSE_create()
#if tracker_type == "CSRT":
#    tracker = cv2.TrackerCSRT_create()
    
    # current positon and position in the previous frames are saved
    # save the current position in the list: position_x and  position_y
    # save the current frame number in the list: frame
    #input: 'frame_num' is number of current frame. 'bbox' is 4 point that refers a square that inclose object.  
    def get_save_position(self, frame_num, bbox):
        #when current frame is saved more than two times
        if frame_num== self.frame[-1]:
            self.position_x[-1] = int (bbox[0] + bbox[2]/2)
            self.position_y[-1] = int (bbox[1] + bbox[3]/2)
        # when current frame is saved for first time
        else:
            self.position_x.append(int (bbox[0] + bbox[2]/2))
            self.position_y.append(int (bbox[1] + bbox[3]/2))
            self.frame.append(frame_num)       
            del self.position_x[0]
            del self.position_y[0]
            del self.frame[0]
        # update bbox saved in this class    
        self.bbox = bbox
    
    # save position 
    def save_position(self,frame_num, point):
        #when current frame is saved more than two times
        if frame_num== self.frame[-1]:
            self.position_x[-1] = int (point[0])
            self.position_y[-1] = int (point[1])        
        else:
            self.position_x.append(int (point[0]))
            self.position_y.append(int (point[1]))
            self.frame.append(frame_num)       
            del self.position_x[0]
            del self.position_y[0]
            del self.frame[0]
        
    
    # save fluorescent images
    def save_fl_img(self,fl_frame_num,fl_img):
        self.img_fl=np.append(self.img_fl, fl_img, axis = 0)
        self.frame_fl.append(fl_frame_num)
        if len(self.frame_fl)> particle.MAX_fl_img + 1:
            self.img_fl = np.delete(self.img_fl, list(range(0, 100)), 0) 
            del self.frame_fl[0]
#        print(self.img_fl.shape)
    
    # report missing of object
    def report_missing(self):
         self.appear = self.appear-1
         if self.appear < 0:
             self.appear = 0

    # report that object is found       
    def report_found(self):
         self.appear = self.appear+1
         if self.appear > 10:
             self.appear = 10

    
    # calculate speed of this object based on position in the current and previous frames
    def get_speed(self):
        
        if self.frame[0] > 0:
            #calculate speed based on this frame and a frame captured a few frame before
            self.speed[0]= (self.position_x[-2] - self.position_x[0]) / (self.frame[-2]-self.frame[0])
            self.speed[1]= (self.position_y[-2] - self.position_y[0]) / (self.frame[-2]-self.frame[0])  
        else:
            self.speed[0]=0
            self.speed[1]=0
        return self.speed
    
    
    
    #predict the position of this object in a frame with a number of frame_num
    def positon_speed_predict(self, frame_num):
        speed=self.get_speed()
        if speed[0] > 0 :
            #get movement in x and y
            move_y = (frame_num - self.frame[-2])*self.speed[1]
            move_x = (frame_num - self.frame[-2])*self.speed[0]
            prediction=(self.position_x[-2]+move_x, self.position_y[-2]+move_y, move_x, move_y)
            
        
        else:
            prediction=(-100,-100,-100,-100)
        
        self.kalman_ok = 1   
        return  prediction
    
#    #predict the bbox of this object 
#    def bbox_predict(self, frame_num):
#        pos_tem = self.positon_predict(frame_num)
#        #print(pos_tem)
#        particle.bbox_pred[0]=pos_tem[0]-self.bbox[2]/2
#        particle.bbox_pred[1]=pos_tem[1]-self.bbox[3]/2
#        particle.bbox_pred[2]=self.bbox[2]
#        particle.bbox_pred[3]=self.bbox[3]
#        if particle.bbox_pred[0]<0:
#            particle.bbox_pred[0]=0
#        if particle.bbox_pred[0]+self.bbox[2]>1640:
#            particle.bbox_pred[0]=1640-self.bbox[2]
#        if particle.bbox_pred[1]<0:
#            particle.bbox_pred[1]=0
#        if particle.bbox_pred[1]+self.bbox[3]>1232:
#            particle.bbox_pred[1]=1232-self.bbox[3]
#        return particle.bbox_pred
    
    

    def check_algae(self):
        if self.frame_fl[0] > 0:
            #get the binary mask of first frame
            particle.mask=self.img_fl[1*particle.FL_SIZE:2*particle.FL_SIZE,:]
            ret, particle.mask = cv2.threshold(particle.mask, 30, 255, cv2.THRESH_BINARY)
            for i in range(particle.MAX_fl_img-1):
                #extract square frame
                particle.tem=self.img_fl[(i+2)*particle.FL_SIZE:(i+3)*particle.FL_SIZE,:]
                #get binary mask of this frame
                ret, particle.tem = cv2.threshold(particle.tem, 40, 255, cv2.THRESH_BINARY)
                #update the mask using “AND”
                particle.mask = cv2.bitwise_and(particle.mask,particle.tem)

            if sum(sum(particle.mask))>15*255:
                self.type='algae'
            else:
                self.type='PS_bead'
            
        return self.type
                
    def init_kalman(self):
        # if this particles has been detected and tracted for a few frames
        if self.kalman_init_num == 0:
                #get speed for init of kalman
            if self.frame[0]>0:
                self.speed[0]= (self.position_x[-1] - self.position_x[0]) / (self.frame[-1]-self.frame[0])
                self.speed[1]= (self.position_y[-1] - self.position_y[0]) / (self.frame[-1]-self.frame[0])  
            else:
                self.speed[0]= (self.position_x[-1] - self.position_x[-2]) / (self.frame[-1]-self.frame[-2])
                self.speed[1]= (self.position_y[-1] - self.position_y[-2]) / (self.frame[-1]-self.frame[-2])  
                
            self.kalman = cv2.KalmanFilter(4,2)
            #设置测量矩阵
            self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
            #设置转移矩阵
            self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
            #设置过程噪声协方差矩阵
            self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)*0.003
            #set init position and speed
            self.kalman.statePre=np.array([[self.position_x[-1]],[self.position_y[-1]],[self.speed[0]],[self.speed[1]]],np.float32)
            #correct kalman filter
            self.kalman.correct(np.array([[self.position_x[-1]],[self.position_y[-1]]],np.float32))
            #set flag 
            self.kalman_ok = 1 
        else:
            self.kalman_init_num = self.kalman_init_num-1
    
    #save position and speed prediction. Input is kalman result    
    def save_prediction(self, prediction):
        self.position_prediction[0] =  prediction[0][0]
        self.position_prediction[1] =  prediction[1][0]
        self.speed[0] =  prediction[2][0]
        self.speed[1] =  prediction[3][0]
        
    #save position and speed prediction. Input is predicted by linear     
    def save_prediction_li(self, prediction):
        self.position_prediction[0] =  int(prediction[0])
        self.position_prediction[1] =  int(prediction[1])
        self.speed[0] =  prediction[2]
        self.speed[1] =  prediction[3]
    
    #update tracking 
    #input: 'img_gray' is a gray image. 'frame_num' is num of current frame 
    def tracker_update(self, img_gray,frame_num):
        ok, bbox = self.tracker.update(img_gray)#update tracker
        self.get_save_position(frame_num, bbox)
        # if tracking is not ok, set tracking result as negetive. As a result, tracking is regard to be failed in the following precessing.         
        if not ok:
            self.bbox=(-100,-100,-100,-100)
            self.get_save_position(frame_num, self.bbox)            
        # if speed is abnormal, set tracking result as negetive. As a result, tracking is regard to be failed in the following precessing. 
        if self.frame[-2]>0:
            if abs(self.position_x[-1]-self.position_x[-2])>20 or self.position_y[-1]-self.position_y[-2] < -10:
                self.bbox=(-100,-100,-100,-100)
                self.get_save_position(frame_num, self.bbox)
 
    '''for tracking method by BO Xiong. Find the area which matches the box in a square
    '''
    
    def get_area_tracking(self,box_ps,img_in,length):
        height, width = img_in.shape[:2]
        box_ps_temp = [box_ps[0],box_ps[1]-2,box_ps[2]+length,box_ps[3]+2]
        if box_ps_temp[2]>width:
            box_ps_temp[2] = width
        if box_ps_temp[1]<0:
            box_ps_temp[1] = 0
        if box_ps_temp[3]>height:
            box_ps_temp[3] = height
        img= img_in[box_ps_temp[1]:box_ps_temp[3],box_ps_temp[0]:box_ps_temp[2]]
        return img
    
 
    def tracking(self,template,img_tracking):
        if template.shape[0] > 2:
            if template.shape[0]<img_tracking.shape[0]:
                res = cv2.matchTemplate(img_tracking,template,cv2.TM_CCOEFF_NORMED)  
                loc = np.where( res == res.max())
                return loc[1][0],loc[0][0]-2;
                #get Y offset. loc[0][0]
                #get x offset. loc[1][0]
        else:
            self.tracking_ok = 0
            return 0,0;
            

     
    #init tracking method Matching_in_Square
    def init_tracking_M_I_S(self,box_ps,gray,length):
        self.template = gray[box_ps[1]:box_ps[3],box_ps[0]:box_ps[2]]
        self.box_ps=box_ps
        self.tracking_l=length
        
    def update_tracking_M_I_S_2(self,box_ps_temp,img_in,x_move,y_move):
        height, width = img_in.shape[:2]
        box_ps_temp = [box_ps_temp[0]+x_move,box_ps_temp[1]+y_move,box_ps_temp[2]+x_move,box_ps_temp[3]+y_move]
        if box_ps_temp[0]< 0:
            box_ps_temp[0] = 0
        if box_ps_temp[2]>width:
            box_ps_temp[2] = width
        if box_ps_temp[1]<0:
            box_ps_temp[1] = 0
        if box_ps_temp[3]>height:
            box_ps_temp[3] = height
        img= img_in[box_ps_temp[1]:box_ps_temp[3],box_ps_temp[0]:box_ps_temp[2]]
        return box_ps_temp,img;
        
    #update tracking method Matching_in_Square
    def update_tracking_M_I_S(self,gray,frame_num):
        self.img_tracking = self.get_area_tracking(self.box_ps, gray, self.tracking_l)
        x_move,y_move=self.tracking(self.template,self.img_tracking)
        self.box_ps,self.template = self.update_tracking_M_I_S_2(self.box_ps,gray,x_move,y_move)
        bbox = (self.box_ps[0], self.box_ps[1], self.box_ps[2]-self.box_ps[0], self.box_ps[3]-self.box_ps[1])
        self.get_save_position(frame_num, bbox)        

        
        
        
