# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:44:28 2019

@author: xiong
"""
from particle_class import particle
import cv2
#import dlib
import numpy as np
import glob
import os
#import matplotlib.pyplot as plt
import copy
import math
#import time
import xlwt
  
def calculateDistance(p1,p2):  
     dist = math.sqrt(((p1[0] - p2[0])**2)/9 + (p1[1] - p2[1])**2)  
     return dist  

def draw_bbox (img, bbox, color, thickness):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(img, p1, p2, (color), thickness , 1)
    return img

def bbox2point (bbox):
    position_x=(int (bbox[0] + bbox[2]/2))
    position_y=(int (bbox[1] + bbox[3]/2))
    return position_x, position_y

def create_new_particle(bbox, img_gray, p_List, frame_num):
    #init a particle object
    particle_a= particle(bbox, img_gray)
    #init tracker
    particle_a.tracker_create() 
    ok = particle_a.tracker.init(img_gray, bbox)
    #save position of object
    particle_a.get_save_position(frame_num, bbox)
    #add this particles into the list
    p_List.append(particle_a)

def check_overlap_bbox(bbox_1 , bbox_2):
    overlap_flag = 0
    if bbox_1[2] > 0 and bbox_2[2] > 0:
        x1 = bbox_1[0] + bbox_1[2]/2
        x2 = bbox_2[0] + bbox_2[2]/2
        y1 = bbox_1[1] + bbox_1[3]/2
        y2 = bbox_2[1] + bbox_2[3]/2
        dem=0# demension of overlap area
        # if overlap area is too much
        #calculate the overlapped dem of two box
        if abs(x1-x2)< (bbox_1[2]/2+bbox_2[2]/2):
            if abs(y1-y2)< (bbox_1[3]/2+bbox_2[3]/2):
                dem= (bbox_1[2]/2+bbox_2[2]/2 - abs(x1-x2))*((bbox_1[3]/2+bbox_2[3]/2)-abs(y1-y2))
        # if area is too big, set flag
        if dem>0.5*bbox_1[2]*bbox_1[3]:
            overlap_flag=1
        if dem>0.5*bbox_2[2]*bbox_2[3]:
            overlap_flag=1
    return overlap_flag


def contours2box_ps(contours):
    x2=0
    y2=0
    x1=32000
    y1=32000
    length=len(contours)
    for i in range(length):
        if contours[i][0][0]>x2:
            x2=contours[i][0][0]
        if contours[i][0][0]<x1:
            x1=contours[i][0][0]
        if contours[i][0][1]>y2:
            y2=contours[i][0][1]
        if contours[i][0][1]<y1:
            y1=contours[i][0][1]
    box_ps=[x1,y1,x2,y2]
    #print(box_ps)
    return box_ps

#reading and configuartion
PATH='D:/For-JESS/'
FOLDER_name= 'rbc_origin'
#detector = dlib.simple_object_detector("detector.svm")
D_THOR=10


lensfree_st=[1209,1233,1257,1281,1306,1330,1354,1378,1402,1426,1451,1475,1499]
lensfree_frames = []
for i in range(12):
    lensfree_frames.append(lensfree_st + i*np.ones(13))
lensfree_frames=np.array(lensfree_frames)
lensfree_frames = lensfree_frames.ravel()


#initial 
frame_count=0
frame_num=0
last_frame_num=0
particle_List = []
current_position = np.zeros((2,1),np.float32)
current_prediction = np.zeros((4,1),np.float32)
Distance_1 = 500
Distance_2 = 500
font = cv2.FONT_HERSHEY_SIMPLEX
num_algae=0
num_bead=0
num_detected = 0
fgbg = cv2.createBackgroundSubtractorMOG2()

#creat a excel
book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("summary")
sheet2 = book.add_sheet("note")

    
for f in glob.glob(os.path.join(PATH + FOLDER_name, "*.jpg")):
    frame = cv2.imread(f)
    img_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    last_frame_num=copy.deepcopy(frame_num)
    frame_num= f[-8:-4]
    frame_num = int(frame_num)
    print(frame_num) 
    frame_count=frame_count+1
    f_save=copy.deepcopy(f)
    
#    # if 1st frame, create particles with hog, init tracking
#    if frame_count==1:
#        #detect particles in img
#        dets = detector(img_gray)
#        #init tracker 
#        for k, d in enumerate(dets):
#            #get bbox from dets
#            bbox = (d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top())
#            create_new_particle(bbox, img_gray, particle_List, frame_num)
#            #label particles with square and save
#            draw_bbox (img_gray, bbox, 0, 1)
#            cv2.imwrite(f_save,img_gray) 
#            
#        continue
    

    type_tem='unknown'
        
    '''
    detect particles with hog in img
    '''
#    dets = detector(img_gray)
#    #draw hog detection with black square
#    for k, d in enumerate(dets):
#        #get bbox from dets
#        bbox = (d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top())
#        #draw_bbox (img_gray, bbox, 0,1)
#    #create a arrary 'hog-new' corresponding to each hog detection
#    #0 means this hog detection is new founded
#    #1 means this hog match tracking or kalman prediction
#     hog_new=np.zeros(len(dets),int)
#     

    '''
    detect particles with contour in img
    '''       
    #remove bgd
    fgmask = fgbg.apply(frame)
    
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel) 
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((6,6),np.uint8)
    erosion = cv2.erode(closing,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)

    #detect particles
    image, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hog_new=np.zeros(len(contours),int)-1


    
    #update tracking and update kalman prediction
    num_particle=len(particle_List)
    for j in range(num_particle):
        '''
        update tracker 
        save position in the class article
        save bbox in the class article
        '''
        particle_List[j].tracker_update(img_gray,frame_num)#update tracker
    # remove the bad tracking.     
    # if movement is more than thorshold, set positon to -100,-100,
    for j in range(num_particle):
        if particle_List[j].bbox[0]>-1: #if tracking j is not removed
            for i in range(num_particle-j-1):
                overlap_flag = check_overlap_bbox(particle_List[j].bbox , particle_List[j+i+1].bbox)
                if overlap_flag:
                    particle_List[j].get_save_position(frame_num, (-100,-100,-100,-100))
                    particle_List[j+i+1].get_save_position(frame_num, (-100,-100,-100,-100))
    
        
    for j in range(num_particle):
        '''
        update kalman
        update kalman filter according to the frame_num
        save prediction,i.e, speed and predicted position in the class particle
        Don NOT corrent kalman filter. correct kalman filter in the end of this frame processing
        '''
        #when kalman is available 
        if particle_List[j].kalman_ok:
            for i in range(frame_num-last_frame_num):
                current_prediction = particle_List[j].kalman.predict()
            #save prediction position
            particle_List[j].save_prediction(current_prediction)
            #get distance bwtween predicted position and tracking position
            particle_List[j].distance_k2t=calculateDistance(current_prediction,[particle_List[j].position_x[-1],particle_List[j].position_y[-1]])
            #cv2.circle(img_gray,(current_prediction[0],current_prediction[1]), 25, 255, 1)
        
        
#        '''
#        Check if results conflicted
#        get distance between tracking and hog
#        get distance between kalman predict and hog
#        '''
#        Distance_t2h = 500
#        Distance_k2h = 500
#        i_contours=-1
#        for k in range(len(contours)):
#            #get current position
#            i_contours=i_contours+1
#            box_ps = contours2box_ps(contours[i_contours])
#            bbox = (box_ps[0], box_ps[1], box_ps[2]-box_ps[0], box_ps[3]-box_ps[1])
#            current_position[0],current_position[1]= bbox2point(bbox)
#           
#            # find the min distance between hog and tracker
#            if calculateDistance(current_position,[particle_List[j].position_x[-1],particle_List[j].position_y[-1]]) < Distance_t2h:
#                Distance_t2h=calculateDistance(current_position,[particle_List[j].position_x[-1],particle_List[j].position_y[-1]])
#                particle_List[j].hog_bbox = copy.deepcopy(bbox)
#                particle_List[j].index_t2h = copy.deepcopy(i_contours)
#                
#            # find the min distance between hog and kalman prediction
#            #when kalman is available 
#            if particle_List[j].kalman_ok:
#                if calculateDistance(current_position,particle_List[j].position_prediction) < Distance_k2h:
#                    Distance_k2h=calculateDistance(current_position,particle_List[j].position_prediction) 
#                    particle_List[j].hog_bbox = copy.deepcopy(bbox)
#                    particle_List[j].index_k2h = copy.deepcopy(i_contours)
#        
#        #save min distance in the class
#        particle_List[j].distance_t2h = copy.deepcopy(Distance_t2h)
#        particle_List[j].distance_k2h = copy.deepcopy(Distance_k2h)
    
    #get cost_matrix
    cost_matrix = np.zeros((len(contours),num_particle))+10000
    for j in range(num_particle):
        for i in range(len(contours)):
            #get position of each contours
            box_ps = contours2box_ps(contours[i])
            bbox = (box_ps[0], box_ps[1], box_ps[2]-box_ps[0], box_ps[3]-box_ps[1])
            x_det,y_det= bbox2point(bbox)
            #get position of each particle
            x_track=particle_List[j].position_x[-1]
            y_track=particle_List[j].position_y[-1]
            #save distance in matrix
            cost_matrix[i,j]=(x_det-x_track)**2+(y_det-y_track)**2
            
    #get cost_matrix using prediction
    cost_matrix_predict = np.zeros((len(contours),num_particle))+10000
    for j in range(num_particle):
        if particle_List[j].kalman_ok:
            for i in range(len(contours)):
                #get position of each contours
                box_ps = contours2box_ps(contours[i])
                bbox = (box_ps[0], box_ps[1], box_ps[2]-box_ps[0], box_ps[3]-box_ps[1])
                x_det,y_det= bbox2point(bbox)
                #get position of each particle
                x_prediction=particle_List[j].position_prediction[0]
                y_prediction=particle_List[j].position_prediction[1]
                #save distance in matrix
                cost_matrix_predict[i,j]=(x_det-x_prediction)**2+(y_det-y_prediction)**2
    
    
    #assignDetectionsToTracks(costMatrix,costOfNonAssignment)
    hog_new=np.zeros(len(contours),int)-1
    CostOfNonAssignment=100
    for j in range(num_particle):
        minElement = np.amin(cost_matrix[:,j])
        hog_i = np.where(cost_matrix[:,j] == minElement)
        hog_i=hog_i[0][0]
        #case assigned tracking 
        if minElement < CostOfNonAssignment: #case 1,2
            #if this particle is not assigned to a det
            if hog_new[hog_i] == -1:
                if particle_List[j].kalman_ok:
                    particle_List[j].case = 1
                else:
                    particle_List[j].case = 2  
                hog_new[hog_i]=j
                particle_List[j].index_2h= hog_i
                    
        #case unassigned tracking
        else: #case 3,4,5,6
            if particle_List[j].kalman_ok: #case 3,4,6 
                x_track=particle_List[j].position_x[-1]
                y_track=particle_List[j].position_y[-1]
                [x_predict,y_predict]= particle_List[j].position_prediction
                distance=(x_predict-x_track)**2+(y_predict-y_track)**2
                
                if distance < CostOfNonAssignment:
                    particle_List[j].case = 4
                else:
                    minElement = np.amin(cost_matrix_predict[:,j])
                    hog_i = np.where(cost_matrix_predict[:,j] == minElement)
                    hog_i=hog_i[0][0]
                    #print(str(particle_List[j].PN)+':'+str(minElement))
                    if minElement < CostOfNonAssignment:
                        particle_List[j].case = 3
                        hog_new[hog_i]=j
                        particle_List[j].index_2h= hog_i
                    else:
                        particle_List[j].case = 6
            else:
                particle_List[j].case = 5
                
     # unassignedDetections= case 3,0
     # case unassigned detecton
#    for i in range(len(contours)):
#        if hog_new[i]==-1:              
#            Distance_k2h = 500
#            #get current det position
#            box_ps = contours2box_ps(contours[i])
#            bbox = (box_ps[0], box_ps[1], box_ps[2]-box_ps[0], box_ps[3]-box_ps[1])
#            for j in range(num_particle):
#                current_position[0],current_position[1]= bbox2point(bbox) 
#                # find the min distance between det and kalman prediction
#                #when kalman is available 
#                if particle_List[j].kalman_ok:
#                    if calculateDistance(current_position,particle_List[j].position_prediction) < Distance_k2h:
#                        Distance_k2h=calculateDistance(current_position,particle_List[j].position_prediction) 
#                        particle_List[j].hog_bbox = copy.deepcopy(bbox)
#                        particle_List[j].index_k2h = copy.deepcopy(i) 
#                        hog_new[i]==j
#            if Distance_k2h < CostOfNonAssignment:#case 3
#                particle_List[hog_new[i]].case = 3
#            
#            else:
#                hog_new[i]=-1#case 0
                
     
    '''
    Deal with found particles
    （1）if detected and tracked matched: add points, correct Kalman
    （2）if detected and tracked matched, kalman not initial: add points, init Kalman
    (3)if detected and tracked not matched, Kalman match detection: correct Kalman, correct tracking
    (4)if detected and tracked not matched, Kalman match tracking: correct Kalman, 
    (5)if detected and tracked not matched, kalman not initial: del this particles
    (6)if detected， and tracked not matched, kalman not match anything: loss points, correct position with Kalman
    （0）Rest of hug detected particles, create new object

    '''
    num_particle=len(particle_List)
    for j in range(num_particle):
        if particle_List[j].case == 1: 
            particle_List[j].report_found()
            #correct kalman filter 
            current_position[0],current_position[1]= bbox2point(particle_List[j].bbox)       
            particle_List[j].kalman.correct(current_position)
#           set number detected
            if particle_List[j].speed[0]>0.5:
                if particle_List[j].appear>4:
                    ok = particle_List[j].assign_PN (num_detected)
                    if ok:
                        num_detected= num_detected + 1
                        
        elif particle_List[j].case == 2: 
            #init kalman filter
            particle_List[j].init_kalman()
            #add point
            particle_List[j].report_found()

        elif particle_List[j].case == 3: 
            #get bbox of target detection
            hog_i = np.where(hog_new == j)
            if len(hog_i[0]):
                hog_i=hog_i[0][0]
                box_ps = contours2box_ps(contours[hog_i])
                bbox = (box_ps[0], box_ps[1], box_ps[2]-box_ps[0], box_ps[3]-box_ps[1])
                #init tracker 
                particle_List[j].tracker_create() 
                ok = particle_List[j].tracker.init(img_gray, bbox)
                particle_List[j].get_save_position(frame_num, bbox)
                #correct kalman filter
                current_position[0],current_position[1]= bbox2point(bbox)       
                particle_List[j].kalman.correct(current_position)
        elif particle_List[j].case == 4:                     
            #correct kalman filter
            current_position[0],current_position[1]= bbox2point(particle_List[j].bbox)       
            particle_List[j].kalman.correct(current_position)
        elif particle_List[j].case == 5:             
            particle_List[j].appear = 0
        elif particle_List[j].case == 6:    
            particle_List[j].report_missing()
            particle_List[j].save_position(frame_num, particle_List[j].position_prediction)

    #check abnormal
    for j in range(num_particle):
        #fix a trouble:stable false tracker
        if particle_List[j].speed[0] < 0.5:
            particle_List[j].report_missing()
        #fix a trouble: tracking bbox keep small
        if  particle_List[j].case==1:         
            #get bbox of target detection
            hog_i = np.where(hog_new == j)
            if len(hog_i[0]):
                hog_i=hog_i[0][0]
                box_ps = contours2box_ps(contours[hog_i])
                bbox = (box_ps[0], box_ps[1], box_ps[2]-box_ps[0], box_ps[3]-box_ps[1])  
                #update tracking bbox with detection
                if bbox[2]*bbox[3]>particle_List[j].bbox[2]*particle_List[j].bbox[3]:
                    #init tracker 
                    particle_List[j].tracker_create() 
                    ok = particle_List[j].tracker.init(img_gray, bbox)
                    particle_List[j].get_save_position(frame_num, bbox)
        #fix a trouble: tracking bbox and detection are working but keep not overlap
        if  particle_List[j].case==1:         
            #get bbox of target detection
            hog_i = np.where(hog_new == j)
            if len(hog_i[0]):
                hog_i=hog_i[0][0]
                box_ps = contours2box_ps(contours[hog_i])
                bbox = (box_ps[0], box_ps[1], box_ps[2]-box_ps[0], box_ps[3]-box_ps[1])  
                #update tracking bbox with detection
                [detection_x,detection_y]=bbox2point(bbox)
                [track_x,track_y]=bbox2point(particle_List[j].bbox)
                if track_x-detection_x>5:
                    print('!!!!!!!!!!!!!!!!!!!!!!!')
            
            
    #remove dead object
    point_p=0
    num_particle=len(particle_List)
    for j in range(num_particle):
        if particle_List[point_p].appear == 0:
            del particle_List[point_p]
        elif particle_List[point_p].position_x[-1] > 820:
            del particle_List[point_p]
        elif particle_List[point_p].speed[0] <-3:
            del particle_List[point_p]
        else:
            point_p=point_p+1
            

            
    
    #case 0
    i_contours=0
    for k in range(len(contours)):
        if hog_new[i_contours]==-1:        
            #get bbox from contours  
            box_ps = contours2box_ps(contours[i_contours])
            bbox = (box_ps[0], box_ps[1], box_ps[2]-box_ps[0], box_ps[3]-box_ps[1])
            create_new_particle(bbox, img_gray, particle_List, frame_num)
            
        i_contours = i_contours + 1


#################################################
    num_particle=len(particle_List)
    for j in range(num_particle):
        if particle_List[j].PN == 3:
            Square=[0,0,0,0]
            Square[0]=int(particle_List[j].bbox[0])
            Square[1]=int(particle_List[j].bbox[0]+particle_List[j].bbox[2])
            Square[2]=int(particle_List[j].bbox[1])
            Square[3]=int(particle_List[j].bbox[1]+particle_List[j].bbox[3])
            print(Square)
            img_bbox=img_gray[Square[2]:Square[3],Square[0]:Square[1]]


#################################### label and summary 
    #print(frame_count)
    num_particle=len(particle_List)
    print('num of tracking object:' + str(num_particle))
    for j in range(num_particle):
#        ok = particle_List[j].assign_PN (num_detected)
#        if ok:
#            num_detected= num_detected + 1

        #write data in excel         
        if particle_List[j].PN > 0:
            sheet1.write(frame_count,2*particle_List[j].PN,particle_List[j].position_x[-1])
            sheet1.write(frame_count,2*particle_List[j].PN+1,particle_List[j].position_y[-1])
        
            
#        if particle_List[j].appear > 0:
            #cv2.circle(img_gray,tuple(particle_List[j].position_prediction), 7, 0, 1)
            #draw_bbox (img_gray, particle_List[j].bbox, 0,1)
            #cv2.putText(img_gray, str(particle_List[j].case),tuple([particle_List[j].position_x[-1],particle_List[j].position_y[-1]]), font, 1, (255), 1, cv2.LINE_AA)
            #if particle_List[j].PN>0:
                #draw_bbox (img_gray, particle_List[j].bbox, 0,1)
                #cv2.putText(img_gray, str(particle_List[j].PN), tuple([particle_List[j].position_x[-1],particle_List[j].position_y[-1]]), font, 1, (255), 1, cv2.LINE_AA)
#        cv2.putText(img_gray, str(j), tuple([particle_List[j].position_x[-1],particle_List[j].position_y[-1]]), font, 1, (255), 1, cv2.LINE_AA)
            #print(str(j) + '_t2h:' + str(particle_List[j].distance_t2h))
            #print(str(j) + '_k2t:' + str(particle_List[j].distance_k2t))

    #show summary
#    cv2.putText(img_gray, 'algae num:' + str(num_algae) ,(100,950), font, 3, (0), 3, cv2.LINE_AA)
#    cv2.putText(img_gray, 'bead num:' + str(num_bead) ,(100,1150), font, 3, (0), 3, cv2.LINE_AA)
#    cv2.putText(img_gray, 'particles_detected:' + str(num_detected) ,(100,100), font, 1, (0), 1, cv2.LINE_AA)
#    img_gray = cv2.drawContours(img_gray, contours, -1, 255, 1)
#    cv2.imwrite(f_save,img_gray) 
    
    #add label in the gray image
    img_color=cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for j in range(num_particle):
        if particle_List[j].PN==3:
            draw_bbox (img_color, particle_List[j].bbox, (255,0,0),2)
        if particle_List[j].PN==4:
            draw_bbox (img_color, particle_List[j].bbox, (0,255,0),2) 
        if particle_List[j].PN==5:
            draw_bbox (img_color, particle_List[j].bbox, (0,0,255),2)

    cv2.imwrite(f_save,img_color) 
    #cv2.imshow('1',img_color)
    print('algae num:' + str(num_algae))
    print('bead num:' + str(num_bead))
    print('all num:' + str(num_detected))
    

#print(dir(particle_a))
    

#############write in title excel
for j in range (num_detected):
    sheet1.write(0,2*j,str(j)+':x')
    sheet1.write(0,2*j+1,str(j)+':y')

book.save("1.xls")
