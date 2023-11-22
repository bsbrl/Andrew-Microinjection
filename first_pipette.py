# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:42:28 2021

@author: enet-joshi317-admin
"""

from XYZ_Stage.XYZ_Position import XYZ_Location
import cv2 
from ML.ml_injection_point_estimation_new import ml_injection_point_estimation_new
import time
from Delta_XY_FOV_2_lin import Delta_XY_FOV_2_lin
from Delta_XY_FOV_1_lin import Delta_XY_FOV_1_lin
import numpy as np
# from stream_image import stream_image
from queue_image import queue_image
from Pressure_Control.Continuous_Pressure import continuous_pressure

def first_pipette(view_1_x,view_1_y,view_2_x,view_2_y,X_pos,Y_pos,Z_pos,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,ser,Z_initial,pic,arduino,q):

    X_est=X_pos
    Y_est=Y_pos
    Z_est=Z_pos
    Z_est_old=Z_pos
    view_1_x_old=view_1_x
    view_2_x_old=view_2_x
    
    print('Current X = ',X_est)
    print('Current Y = ',Y_est)
    print('Current Z = ',Z_est)
    print('New Pipette')
    print('TURN VALVES!')
    #Original robot
    XYZ_Location(11250,11250,8000,32340,31000,0,ser)
    #Prototpye robot
    # XYZ_Location(11250,11250,8000,60000,60000,5000,ser)
    time.sleep(10)

    # print('Focus first time')
    # #uncommented
    # pressure_value=2
    # correct=0
    # o=0
    # while correct==0:
    #     print('Try ',o+1)
    #     signal=continuous_pressure(0,pressure_value,'inj')
    #     arduino.write(signal.encode())
    #     arduino.flush()
    #     q_=arduino.readline()
    #     q_=q_.decode()
    #     s=q_.find('Received')
    #     if q_[s+9]=='P' and q_[s+10:s+12]==str(int((pressure_value + 0.35)/0.107)) and q_[s+12]=='p' and q_[s+13]=='\r':
    #         correct=1
    #     else:
    #         o+=1 
    # #uncommented
    # print('OPEN PIPETTE')
    # time.sleep(15)
    # correct=0
    # o=0
    # while correct==0:
    #     print('Try ',o+1)
    #     signal=continuous_pressure(0,pressure_value,'inj')
    #     arduino.write("P0p".encode())
    #     arduino.flush()
    #     q_=arduino.readline()
    #     q_=q_.decode()
    #     s=q_.find('Received')
    #     if q_[s+9]=='P' and q_[s+10]=='0' and q_[s+11]=='p' and q_[s+12]=='\r':
    #         correct=1
    #     else:
    #         o+=1
            
    img1,img2=queue_image(q)
    im_height_1=720
    im_width_1=1280
    im_height_2=720
    im_width_2=1280
    
    lower_blue = np.array([52,30,35])
    upper_blue = np.array([255,255,255])
    output_dict_detection_boxes_stored_pipette_1,output_dict_detection_classes_stored_pipette_1,output_dict_detection_scores_stored_pipette_1,y1a_rc_pipette_1,y2a_rc_pipette_1,x1a_rc_pipette_1,x2a_rc_pipette_1,xc_rc_pipette_1,yc_rc_pipette_1=ml_injection_point_estimation_new([img1],.01,720,1280,graph,sess,1)
    output_dict_detection_boxes_stored_pipette_2,output_dict_detection_classes_stored_pipette_2,output_dict_detection_scores_stored_pipette_2,y1a_rc_pipette_2,y2a_rc_pipette_2,x1a_rc_pipette_2,x2a_rc_pipette_2,xc_rc_pipette_2,yc_rc_pipette_2=ml_injection_point_estimation_new([img2],.01,720,1280,graph,sess,1)
    list_classes_pipette_1=output_dict_detection_classes_stored_pipette_1[0].tolist()
    if 5 in list_classes_pipette_1:
        list_classes_index_pipette_1=list_classes_pipette_1.index(5)
        crop_1=img1[y1a_rc_pipette_1[0][list_classes_index_pipette_1]:y2a_rc_pipette_1[0][list_classes_index_pipette_1],x1a_rc_pipette_1[0][list_classes_index_pipette_1]:x2a_rc_pipette_1[0][list_classes_index_pipette_1]]
        hsv_1 = cv2.cvtColor(crop_1, cv2.COLOR_BGR2HSV)
        mask_1 = cv2.inRange(hsv_1, lower_blue, upper_blue)
        mask_1_list=mask_1.tolist()
        x_list=[]
        y_list=[]
        for j in range(len(mask_1)):
            indices = [i for i, x in enumerate(mask_1_list[j]) if x==255]
            if indices!=[]:
                x_list.append(np.median(indices))
                y_list.append(j)
        if x_list==[] or y_list==[] or x1a_rc_pipette_1==[] or y1a_rc_pipette_1==[]:
            view_1_x=int(xc_rc_pipette_1[0][list_classes_index_pipette_1])
            view_1_y=int(yc_rc_pipette_1[0][list_classes_index_pipette_1])
            print('CV tip x = ',view_1_x)
            print('CV tip y = ',view_1_y)
        else:
            view_1_x=int(x_list[len(x_list)-1]+x1a_rc_pipette_1[0][list_classes_index_pipette_1])
            view_1_y=int(y_list[len(y_list)-1]+y1a_rc_pipette_1[0][list_classes_index_pipette_1])
            print('CV tip x = ',int(x_list[len(x_list)-1]+x1a_rc_pipette_1[0][list_classes_index_pipette_1]))
            print('CV tip y = ',int(y_list[len(y_list)-1]+y1a_rc_pipette_1[0][list_classes_index_pipette_1]))
    list_classes_pipette_2=output_dict_detection_classes_stored_pipette_2[0].tolist()
    if 5 in list_classes_pipette_2:
        list_classes_index_pipette_2=list_classes_pipette_2.index(5)
        crop_2=img2[y1a_rc_pipette_2[0][list_classes_index_pipette_2]:y2a_rc_pipette_2[0][list_classes_index_pipette_2],x1a_rc_pipette_2[0][list_classes_index_pipette_2]:x2a_rc_pipette_2[0][list_classes_index_pipette_2]]
        hsv_2 = cv2.cvtColor(crop_2, cv2.COLOR_BGR2HSV)
        mask_2 = cv2.inRange(hsv_2, lower_blue, upper_blue)
        mask_2_list=mask_2.tolist()
        x_list=[]
        y_list=[]
        for j in range(len(mask_2)):
            indices = [i for i, x in enumerate(mask_2_list[j]) if x==255]
            if indices!=[]:
                x_list.append(np.median(indices))
                y_list.append(j)
        if x_list==[] or y_list==[] or x1a_rc_pipette_2==[] or y1a_rc_pipette_2==[]:
            view_2_x=int(xc_rc_pipette_2[0][list_classes_index_pipette_2])
            view_2_y=int(yc_rc_pipette_2[0][list_classes_index_pipette_2]) 
            print('CV tip x = ',view_2_x)
            print('CV tip y = ',view_2_y)
        else:
            view_2_x=int(x_list[len(x_list)-1]+x1a_rc_pipette_2[0][list_classes_index_pipette_2])
            view_2_y=int(y_list[len(y_list)-1]+y1a_rc_pipette_2[0][list_classes_index_pipette_2])
            print('CV tip x = ',int(x_list[len(x_list)-1]+x1a_rc_pipette_2[0][list_classes_index_pipette_2]))
            print('CV tip y = ',int(y_list[len(y_list)-1]+y1a_rc_pipette_2[0][list_classes_index_pipette_2]))
    
    # original
    x1_1_crop=int(view_1_x-400)
    x2_1_crop=int(view_1_x+400)
    y1_1_crop=int(view_1_y-100)
    y2_1_crop=int(view_1_y+300)
       

    x1_2_crop=int(view_2_x-400)
    x2_2_crop=int(view_2_x+400)
    y1_2_crop=int(view_2_y-100)
    y2_2_crop=int(view_2_y+300)

    # # prototype
    # x1_1_crop=int(view_1_x-600)
    # x2_1_crop=int(view_1_x+600)
    # y1_1_crop=int(view_1_y-300)
    # y2_1_crop=int(view_1_y+300)
       

    # x1_2_crop=int(view_2_x-600)
    # x2_2_crop=int(view_2_x+600)
    # y1_2_crop=int(view_2_y-300)
    # y2_2_crop=int(view_2_y+300)
    
    if x1_1_crop<0:
        x1_1_crop=0
    if y1_1_crop<0:
        y1_1_crop=0
    if x2_1_crop>1280:
        x2_1_crop=1280    
    if y2_1_crop>720:
        y2_1_crop=720
        
    if x1_2_crop<0:
        x1_2_crop=0
    if y1_2_crop<0:
        y1_2_crop=0
    if x2_2_crop>1280:
        x2_2_crop=1280
    if y2_2_crop>720:
        y2_2_crop=720
        
    im_width_1=x2_1_crop-x1_1_crop
    im_height_1=y2_1_crop-y1_1_crop 
    im_width_2=x2_2_crop-x1_2_crop
    im_height_2=y2_2_crop-y1_2_crop
    print(x1_2_crop)
    print(x2_2_crop)
    print(y1_2_crop)
    print(y2_2_crop)

    s_end=0
    s=0
    dx_p_1,dy_p_1=Delta_XY_FOV_1_lin(view_1_x,view_1_x_old)
    dx_p_2,dy_p_2=Delta_XY_FOV_2_lin(view_2_x,view_2_x_old)
    dx_p=dx_p_1+dx_p_2
    dy_p=dy_p_1+dy_p_2
    X_est=X_est+dx_p_1+dx_p_2
    Y_est=Y_est+dy_p_1+dy_p_2
    XYZ_Location(5000,5000,2000,X_est,Y_est,Z_est,ser)
    time.sleep(10)
    
    while s_end==0 and Z_est<Z_initial+500:
        Z_est=Z_est_old+100*s
        XYZ_Location(5000,5000,2000,X_est,Y_est,Z_est,ser)
        time.sleep(.5)
        # img1,img2=stream_image(footage_socket_1,footage_socket_2,footage_socket_3,pic,0)
        img1,img2=queue_image(q)
        img1_crop=img1[y1_1_crop:y2_1_crop,x1_1_crop:x2_1_crop]
        img2_crop=img2[y1_2_crop:y2_2_crop,x1_2_crop:x2_2_crop]
        output_dict_detection_boxes_stored_1,output_dict_detection_classes_stored_1,output_dict_detection_scores_stored_1,y1a_rc_1,y2a_rc_1,x1a_rc_1,x2a_rc_1,xc_rc_1,yc_rc_1=ml_injection_point_estimation_new([img1_crop],.1,im_height_1,im_width_1,graph,sess,1)
        list_classes_1_c=output_dict_detection_classes_stored_1[0].tolist()
        output_dict_detection_boxes_stored_2,output_dict_detection_classes_stored_2,output_dict_detection_scores_stored_2,y1a_rc_2,y2a_rc_2,x1a_rc_2,x2a_rc_2,xc_rc_2,yc_rc_2=ml_injection_point_estimation_new([img2_crop],.1,im_height_2,im_width_2,graph,sess,1)
        list_classes_2_c=output_dict_detection_classes_stored_2[0].tolist()
        if 1 not in list_classes_1_c or 1 not in list_classes_2_c:
            print('Centroid not detected in both FOVs')   
            s+=1       
        else:
            print('Embryo in FOV')
            print('Initial Z estimate = ',Z_est+100)
            Z_new=Z_est+100
            XYZ_Location(5000,5000,2000,X_est,Y_est,Z_new,ser)
            time.sleep(.5)
            s_end=1
                
    return footage_socket_1,footage_socket_2,Z_new,dx_p,dy_p,view_1_x,view_1_y,view_2_x,view_2_y
