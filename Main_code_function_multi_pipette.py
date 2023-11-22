# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 12:53:33 2023

@author: User
"""

def stream_function_multi_process(q):
    import base64
    import cv2
    import zmq
    import numpy as np
    
    context = zmq.Context()
    footage_socket_1 = context.socket(zmq.PUB)
    footage_socket_1.setsockopt(zmq.LINGER, 0)
    footage_socket_1.connect('tcp://localhost:5555')
    footage_socket_2 = context.socket(zmq.PUB)
    footage_socket_2.setsockopt(zmq.LINGER, 0)
    footage_socket_2.connect('tcp://localhost:4555')
    footage_socket_3 = context.socket(zmq.SUB)
    try:
        footage_socket_3.bind('tcp://*:3555')
    except:
        print('socket already in use')
    footage_socket_3.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

    cap_1=cv2.VideoCapture(0)
    cap_2=cv2.VideoCapture(1)
    cap_1.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap_1.set(cv2.CAP_PROP_FPS,30)
    cap_1.set(cv2.CAP_PROP_AUTOFOCUS,1)
    cap_2.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap_2.set(cv2.CAP_PROP_FPS,30)
    cap_2.set(cv2.CAP_PROP_AUTOFOCUS,1)
    
    video_on=1
    inj_num=1
    while video_on==1:
        grabbed_1, frame_1 = cap_1.read()  # grab the current frame
        grabbed_2, frame_2 = cap_2.read()  # grab the current frame
        check=footage_socket_3.poll(1)
        if check==1:
            inj_num=footage_socket_3.recv()
            inj_num=int(inj_num.decode('utf-8'))
            encoded_1, buffer_1 = cv2.imencode('.jpg', frame_1)
            encoded_2, buffer_2 = cv2.imencode('.jpg', frame_2)
            jpg_as_text_1 = base64.b64encode(buffer_1)
            jpg_as_text_2 = base64.b64encode(buffer_2)
            rec=0
            while rec!=-1:
                footage_socket_1.send(jpg_as_text_1)
                footage_socket_2.send(jpg_as_text_2)
                check=footage_socket_3.poll(1)
                if check==1:
                    num=footage_socket_3.recv()
                    rec=int(num.decode('utf-8'))
        else:
            inj_num=inj_num
        inj_num=str(inj_num)
        cv2.putText(frame_1, 'Embryo '+inj_num, (70,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2,cv2.LINE_AA)
        cv2.putText(frame_2, 'Embryo '+inj_num, (70,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2,cv2.LINE_AA)
        frame_1 = cv2.resize(frame_1, (889, 500))  # resize the frame
        frame_2 = cv2.resize(frame_2, (889, 500))  # resize the frame
        if q.empty()==True:
            q.put([0,frame_1,frame_2])
    cap_1.release()
    cap_2.release()
    footage_socket_1.close()
    footage_socket_2.close()
    footage_socket_3.close()

def stream_function_multi_process_new(q):
    import cv2

    cap_1=cv2.VideoCapture(0)
    cap_2=cv2.VideoCapture(1)
    cap_1.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap_1.set(cv2.CAP_PROP_FPS,60)
    cap_1.set(cv2.CAP_PROP_AUTOFOCUS,1)
    cap_2.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap_2.set(cv2.CAP_PROP_FPS,60)
    cap_2.set(cv2.CAP_PROP_AUTOFOCUS,1)
    
    video_on=1
    while video_on==1:
        grabbed_1, frame_1 = cap_1.read()  # grab the current frame
        grabbed_2, frame_2 = cap_2.read()  # grab the current frame
        if q.empty()==True:
            q.put([0,frame_1,frame_2])
    cap_1.release()
    cap_2.release()

def Change_pipette(pipette_number,r):
    import time
    r.put([pipette_number])
    time.sleep(1)
    
            
def Main_code(target_pixel,inj_depth,inj_speed,path,pipette_number,q,r):
    # import os
    from XYZ_Stage.XYZ_Position import XYZ_Location
    from DSLR_Camera.DSLR_Call import func_TakeNikonPicture
    # from ML.ml_whole_image import ml
    from ML.process_whole_detections import process_whole_detections
    from ML.order import order 
    import time
    import numpy as np
    from injection_ml_tip_short_new_thresh import injection_ml_tip_short_new_thresh
    from new_pipette_new import new_pipette_new
    from first_pipette import first_pipette
    from path_finder_new import path_finder_new
    from ML.detections_dslr_image import detections_dslr_image
    import tensorflow as tf
    import math
    import serial
    from ML.transformation_matrix_DSLR_pipette import function_transformation_matrix_DSLR_pipette
    from new_z import new_z
    from injection_results_new import injection_results_new
    import cv2
    
    # os.system("taskkill /im python.exe /F")

    total_start_time=time.time()
    receive=r.get()
    dish_num=str(receive[0])
    target_pixel=int(target_pixel)
    inj_depth=int(inj_depth)
    inj_speed=int(inj_speed)
    pipette_number=int(receive[1])
    # Initial Variables
    # small dish height and light should be 2 intensity, 6 for regular
    # z_needle=19000
    # z_needle=16000
    # z_needle=9000
    z_needle=7000
    Z_initial=z_needle+5000      
    width_image=6000
    height_image=4000
    thresh_ml=.1
    sum_image_thresh_max=20000
    sum_image_thresh_min=1500
    over_injected=0
    missed=0
    inj_num=0
    over_injected=0
    missed=0
    no_injected=0
    inj_num_init=inj_num
    pressure_value=5
    # pressure_value=1
    pressure_value_current=1
    # pressure_value=2
    back_pressure_value=5
    pressure_time=4
    post_z=-200
    pipette=1
    calib_pipette_num=1
    pip_num=0
    pip_em_num=[0]
    switch_list=[]
    injected_embryos=0
    injected_embryos_count=0
    injected=2
    miss=1
    injection_list=[]
    injected_list=[]
    elim_embryo=[]
    deltas_pipette=[[0,0,0]]
    injection_time_list=[]
    inj_time_total=0
    footage_socket_1=0
    footage_socket_2=0
    footage_socket_3=0
    inj_pressure=0
    # needle_tfs=[[44203,30200,56430,3600,22280,11950],[44130,30200,56330,3550,22180,11950]]
    # needle_pxs=[[621,325,588,356],[730,353,722,242]]
    # needle_tfs=[[42380,35750,66030,6900,19220,9000],[42430,35700,66050,6850,19270,8950]]
    # needle_pxs=[[403,293,706,281],[515,343,775,405]]

    needle_tfs=[[40080,31250,56380,10050,24330,13450],[40080,31250,56380,10050,24330,13450]]
    needle_pxs=[[593,206,782,215],[593,206,782,215]]
    
    # Connect XYZ stage
    try:
      ser = serial.Serial(
        port="COM3",
        baudrate=9600,
      )
      ser.isOpen() # try to open port, if possible print message and proceed with 'while True:'
      print ("port is opened!")
    
    except IOError: # if port is already opened, close it and open it again and print message
      ser.close()
      ser.open()
      print ("port was already open, was closed and opened again!")


    # Connect to arduino
    try:
      arduino = serial.Serial(
        port="COM7",
        baudrate=9600,
      )
      arduino.isOpen() # try to open port, if possible print message and proceed with 'while True:'
      print ("port is opened!")
    
    except IOError: # if port is already opened, close it and open it again and print message
      arduino.close()
      arduino.open()
      print ("port was already open, was closed and opened again!")
    
    time.sleep(5)
    print('Connecting to arduino')

    # Go to camera 
    print('Moving under DSLR')
    # take picture
    filename='Entire_Petri_Dish_'+dish_num+'.jpg'
    XYZ_Location(10000,10000,8000,54430,93000,0,ser)
    time.sleep(10)
    XYZ_Location(10000,10000,8000,54430,93000,5000,ser)
    q.put([3])
    r.put([0,0,0,54430,93000,5000,'No',0,0])
    time.sleep(20)
    func_TakeNikonPicture(filename)
    time.sleep(10)
    image=cv2.imread(path+'/DSLR_Camera/'+filename)
    image_height=6000
    image_width=4000
    center=(int((float(image_width))/(2)),int((float(image_height))/(2)))
    scale=1
    M_1 = cv2.getRotationMatrix2D(center,270, scale)
    cosine = np.abs(M_1[0, 0])
    sine = np.abs(M_1[0, 1])
    nW = int((image_height * sine) + (image_width * cosine))
    nH = int((image_height * cosine) + (image_width * sine))
    M_1[0, 2] += (nW / 2) - int((float(image_width))/(2))
    M_1[1, 2] += (nH / 2) - int((float(image_height))/(2))
    image=cv2.warpAffine(image, M_1, (image_height, image_width)) 
    cv2.imwrite(path+'/DSLR_Camera/'+filename,image)
    image=image[450:3485,510:3545]
    img_dish_gui=cv2.resize(image,(500,500))
    cv2.imwrite(path+'/DSLR_Camera/'+'gui_'+filename,img_dish_gui)
    q.put([1])
    r.put([img_dish_gui])

    # detect embryos
    print('Detecting embryos')
    # xc_rc,yc_rc,scores=ml(path+'/faster_r_cnn_trained_model_petri_new_8',path+'/DSLR_Camera/'+filename,thresh_ml,width_image,height_image,tf,np)
    
    # Open new tensorflow session
    graph = tf.Graph()
    with graph.as_default():
      od_graph_def = tf.compat.v1.GraphDef()
      with tf.compat.v2.io.gfile.GFile(path+'/faster_r_cnn_trained_model_petri_new_8'+'/frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            xc_rc,yc_rc=process_whole_detections(path,path+'/faster_r_cnn_trained_model_petri_new_8',path+'/DSLR_Camera/'+filename,width_image,height_image,thresh_ml,tf,np,graph,sess)
    
    img_dish=cv2.imread(path+'/DSLR_Camera/'+filename,1)
    y1a_rc,y2a_rc,x1a_rc,x2a_rc=order(xc_rc,yc_rc,0,0)

    for i in range(len(y1a_rc)):
        cv2.rectangle(img_dish,(int(x1a_rc[i][0]),int(y1a_rc[i][0])),(int(x2a_rc[i][0]),int(y2a_rc[i][0])),(0,255,0),3)
    cv2.imwrite(path+'/ML_and_injections/img.jpg',img_dish)
    img_dish=img_dish[450:3485,510:3545]
    img_dish_gui=cv2.resize(img_dish,(500,500))

    xc_rc_new_list=[]
    yc_rc_new_list=[]
    x1a_rc_new_list=[]
    y1a_rc_new_list=[]
    x2a_rc_new_list=[]
    y2a_rc_new_list=[]
    for i in range(len(xc_rc)):
        xc_rc_new_list.append(int(xc_rc[i][0]))
        yc_rc_new_list.append(int(yc_rc[i][0]))
        x1a_rc_new_list.append(int(xc_rc[i][1]))
        y1a_rc_new_list.append(int(yc_rc[i][1]))
        x2a_rc_new_list.append(int(xc_rc[i][2]))
        y2a_rc_new_list.append(int(yc_rc[i][2]))
    cv2.imwrite(path+'/DSLR_Camera/ML_Petri_Dishes/'+filename,img_dish)
    cv2.imwrite(path+'/Video images/ML_image/'+filename,img_dish)
    
    cv2.imwrite(path+'/DSLR_Camera/ML_Petri_Dishes/'+'gui_'+filename,img_dish_gui)
    q.put([2])
    r.put([img_dish_gui])
    print('Finished detecting embryos')
    print('Number of embryos = {}'.format(len(xc_rc)))
    time.sleep(5)
    q.put([0])

    # Open new tensorflow session
    graph = tf.Graph()
    with graph.as_default():
      od_graph_def = tf.compat.v1.GraphDef()
      with tf.compat.v2.io.gfile.GFile(path+'/faster_r_cnn_trained_model_injection_point_tip_pipette_robot_2_new_4'+'/frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # img_dish=cv2.imread(path+'/DSLR_Camera/'+filename,1)
            img_dish=cv2.imread(path+'/ML_and_injections/img.jpg',1)
            x1a_rc_post,y1a_rc_post,x2a_rc_post,y2a_rc_post,xc_rc_keep,yc_rc_keep=path_finder_new(xc_rc_new_list,yc_rc_new_list,x1a_rc_new_list,y1a_rc_new_list,x2a_rc_new_list,y2a_rc_new_list,img_dish,filename)
            img_path=cv2.imread(path+'/Video images/Path_image/'+filename)
            img_path=img_path[450:3485,510:3545]
            img_path_gui=cv2.resize(img_path,(500,500))
            cv2.imwrite(path+'/Video images/Path_image/'+'gui_'+filename,img_path_gui)
            q.put([1])
            r.put([img_path_gui])
            
            positions=[]
            for i in range(len(xc_rc_keep)):
                embryo_point_center = function_transformation_matrix_DSLR_pipette(xc_rc_keep[i],yc_rc_keep[i],2062,1130,1293,2132,2805,1971,needle_tfs[pipette_number-1][0],needle_tfs[pipette_number-1][1],needle_tfs[pipette_number-1][2],needle_tfs[pipette_number-1][3],needle_tfs[pipette_number-1][4],needle_tfs[pipette_number-1][5]) # DSLR to pipette
                positions.append([int(float(embryo_point_center.item(0,0))),int(float(embryo_point_center.item(1,0)))])
                # positions.append([int(float(embryo_point_center.item(0,0)))-550,int(float(embryo_point_center.item(1,0)))-250])
            print(len(positions))
            
            for pic in range(len(positions)):
                total_start_time_inj=time.time()
                print('Embryo {} out of {} Embryos'.format(pic+1,len(positions)))
                cv2.rectangle(img_dish, (xc_rc_keep[pic]-20, yc_rc_keep[pic]-20), (xc_rc_keep[pic]+20, yc_rc_keep[pic]+20), (0, 0, 0), thickness=3)
                img_dish_small=img_dish[450:3485,510:3545]
                img_path_gui=cv2.resize(img_dish_small,(500,500))
                q.put([1])
                r.put([img_path_gui])
                if pic==0:
                    print('New X = ',positions[pic][0])
                    print('New Y = ',positions[pic][1])
                    print('New Z = ',z_needle-300)
                    dx_final=0
                    dy_final=0
                    dz=0
                    view_1_x=needle_pxs[pipette_number-1][0]
                    view_1_y=needle_pxs[pipette_number-1][1]
                    view_2_x=needle_pxs[pipette_number-1][2]
                    view_2_y=needle_pxs[pipette_number-1][3]
                    time_wait=4.1
                    print('Start air pressure')
                    print('Start stream')
                    q.put([3])
                    r.put([len(positions),pic+1,injected_embryos,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle,'No',int(inj_time_total/60),int(100*(injected_embryos/len(positions)))])
                    footage_socket_1,footage_socket_2,z_needle_new,dx_final,dy_final,view_1_x,view_1_y,view_2_x,view_2_y=first_pipette(view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,ser,Z_initial,pic,arduino,q)
                    q.put([3])
                    r.put([len(positions),pic+1,injected_embryos,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle,'No',int(inj_time_total/60),int(100*(injected_embryos/len(positions)))])
                    dx_final,dy_final,X_inj,Y_inj,Z_inj,footage_socket_1,footage_socket_2,injection_list_num,Z_new,dz,view_1_x,view_1_y,view_2_x,view_2_y,Z_inj_actual,pipette,current_x_centroid,current_y_centroid,current_z_centroid,injected,sum_image,pressure_value_current,injection_time,miss,inj_pressure=injection_ml_tip_short_new_thresh(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle_new,dx_final,dy_final,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,pic,sum_image_thresh_min,target_pixel,miss,q,inj_pressure,pressure_value_current)
                else:
                    print('New X = ',positions[pic][0])
                    print('New Y = ',positions[pic][1])
                    print('New Z = ',Z_new )
                    dist=math.hypot(positions[pic-1][0]-positions[pic][0],positions[pic-1][1]-positions[pic][1])
                    cv2.line(img_dish, (xc_rc_keep[pic-1], yc_rc_keep[pic-1]), (xc_rc_keep[pic], yc_rc_keep[pic]), (0, 125, 0), thickness=3)
                    print('Distance traveled = ',dist)
                    if dist>12000:
                        Z_new=new_z(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle,dx_final,dy_final,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,ser,pip_num,Z_inj_actual,pic,q)
                        miss=1
                    q.put([3])
                    r.put([len(positions),pic+1,injected_embryos,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),Z_new,'No',int(inj_time_total/60),int(100*(injected_embryos/len(positions)))])
                    dx_final,dy_final,X_inj,Y_inj,Z_inj,footage_socket_1,footage_socket_2,injection_list_num,Z_new,dz,view_1_x,view_1_y,view_2_x,view_2_y,Z_inj_actual,pipette,current_x_centroid,current_y_centroid,current_z_centroid,injected,sum_image,pressure_value_current,injection_time,miss,inj_pressure=injection_ml_tip_short_new_thresh(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),Z_new,dx_final,dy_final,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,pic,sum_image_thresh_min,target_pixel,miss,q,inj_pressure,pressure_value_current)
                np.save('tip.npy',np.array([view_1_x,view_1_y,view_2_x,view_2_y]))
                inj_num+=1
                pipette=0
                
                if injection_time!=0:
                    injection_time_list.append(injection_time)
                if -10<int(sum_image)<sum_image_thresh_min:
                    switch_list.append(1)
                elif int(sum_image)>=sum_image_thresh_min:
                    switch_list.append(0)
                else:
                    print('append nothing')
                if injected==2:
                    # inj_pressure=0
                    missed+=1
                    print('Missed injection')
                    elim_embryo.append([current_x_centroid,current_y_centroid,current_z_centroid,pip_num,pic])
                    print('Number of injected embryos = ',injected_embryos)
                    print('Remove embryo from dish')
                    print('Number of embryos missed = ',missed)
                    injection_list.append(4) 
                    cv2.rectangle(img_dish, (xc_rc_keep[pic]-20, yc_rc_keep[pic]-20), (xc_rc_keep[pic]+20, yc_rc_keep[pic]+20), (0,0,255), thickness=3)
                elif int(sum_image)<sum_image_thresh_min:
                    no_injected+=1
                    print('No injection')
                    elim_embryo.append([current_x_centroid,current_y_centroid,current_z_centroid,pip_num,pic])
                    print('Number of injected embryos = ',injected_embryos)
                    print('Remove embryo from dish')
                    print('Number of embryos not injected = ',no_injected)
                    injection_list.append(4) 
                    injected_list.append(0) 
                    cv2.rectangle(img_dish, (xc_rc_keep[pic]-20, yc_rc_keep[pic]-20), (xc_rc_keep[pic]+20, yc_rc_keep[pic]+20), (0,0,255), thickness=3)
                else:
                    # inj_pressure=1
                    if sum_image>sum_image_thresh_max and injected!=2 and int(sum_image)!=0:
                        over_injected+=1
                        pressure_time=4
                        injected_embryos+=1
                        injected_embryos_count+=1
                        print('Number of injected embryos = ',injected_embryos)
                        print('Remove embryo from dish')
                        print('Number of embryos over injected = ',over_injected)
                        injection_list.append(4) 
                        injected_list.append(1) 
                        cv2.rectangle(img_dish, (xc_rc_keep[pic]-20, yc_rc_keep[pic]-20), (xc_rc_keep[pic]+20, yc_rc_keep[pic]+20), (171,118,55), thickness=3)
                    elif sum_image_thresh_min<sum_image<sum_image_thresh_max and injected!=2 and int(sum_image)!=0:
                        injection_list.append(injection_list_num)
                        print('Successful injection')
                        injected_embryos+=1
                        injected_embryos_count+=1
                        print('Number of injected embryos = ',injected_embryos)
                        injected_list.append(1)
                        cv2.rectangle(img_dish, (xc_rc_keep[pic]-20, yc_rc_keep[pic]-20), (xc_rc_keep[pic]+20, yc_rc_keep[pic]+20), (171,118,55), thickness=3)
                    else:
                        print('Good')
                        cv2.rectangle(img_dish, (xc_rc_keep[pic]-20, yc_rc_keep[pic]-20), (xc_rc_keep[pic]+20, yc_rc_keep[pic]+20), (171,118,55), thickness=3)
                total_end_time_inj=time.time()
                inj_time=int(total_end_time_inj-total_start_time_inj)
                inj_time_total+=inj_time
                img_dish_small=img_dish[450:3485,510:3545]
                img_path_gui=cv2.resize(img_dish_small,(500,500))
                q.put([1])
                r.put([img_path_gui])
                q.put([3])
                r.put([len(positions),pic+1,injected_embryos,current_x_centroid,current_y_centroid,current_z_centroid,'No',int(inj_time_total/60),int(100*(injected_embryos/len(positions)))])
                
                # if switch_list[len(switch_list)-1]==1:
                if len(switch_list)>2 and switch_list[len(switch_list)-3]==1 and switch_list[len(switch_list)-2]==1 and switch_list[len(switch_list)-1]==1:
                    print('CHANGE TO NEW PIPETTE AND VALVES!')
                    XYZ_Location(10000,10000,8000,49430,8000,0,ser)
                    # inj_pressure=0
                    q.put([3])
                    r.put([len(positions),pic+1,injected_embryos,current_x_centroid,current_y_centroid,current_z_centroid,'Yes',int(inj_time_total/60),int(100*(injected_embryos/len(positions)))])
                    np.save('currentXY.npy',np.array([positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final)]))
                    r.put([pipette_number])
                    time.sleep(1)
                    pipette_number_new=pipette_number
                    while pipette_number==pipette_number_new:
                        receive=r.get()
                        time.sleep(1)
                        pipette_number_new=int(receive[0])
                    pipette_number=pipette_number_new
                    q.put([3])
                    r.put([len(positions),pic+1,injected_embryos,current_x_centroid,current_y_centroid,current_z_centroid,'No',int(inj_time_total/60),int(100*(injected_embryos/len(positions)))])
                    time.sleep(1)
                    miss=1
                    # pressure_value=1
                    pressure_value=5
                    pressure_time=4
                    time_wait=10
                    injected_list=[]
                    switch_list=[]
                    pipette=1
                    pip_num+=1
                    calib_pipette_num+=1
                    pip_em_num.append(pic+1)
                    dx_final,dy_final,current_x,current_y,current_z,footage_socket_1,footage_socket_2,Z_new,view_1_x,view_1_y,view_2_x,view_2_y,injected_embryos_count,dz_final,current_z_needle=new_pipette_new(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle,dx_final,dy_final,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,ser,pip_num,Z_inj_actual,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,post_z,Z_initial,current_z_centroid,pic,sum_image_thresh_min,target_pixel,q,inj_pressure,pressure_value_current) 
                    deltas_pipette.insert(pip_num-1,[current_x-current_x_centroid,current_y-current_y_centroid,current_z_needle-current_z_centroid])
                    if pic+1<len(positions):
                        XYZ_Location(20000,20000,8000,positions[pic+1][0]+int(dx_final),positions[pic+1][1]+int(dy_final),0,ser)
                        time.sleep(5)
                        XYZ_Location(20000,20000,8000,positions[pic+1][0]+int(dx_final),positions[pic+1][1]+int(dy_final),Z_new,ser)
                        time.sleep(5)
                print('Starting pressure = ',pressure_value)
            # Stop pressure
            time.sleep(1)
            arduino.write("P0p".encode())
            time.sleep(1)
            #eliminate uninjected embryos
            if injected_embryos==len(positions):
                XYZ_Location(20000,20000,8000,current_x,current_y,0,ser)
            else:
                if len(deltas_pipette)<2:
                    print('no adding deltas')
                else:
                    deltas_pipette_x=[]
                    deltas_pipette_y=[]
                    deltas_pipette_z=[]
                    for w in range(len(deltas_pipette)):
                        deltas_pipette_x.append(deltas_pipette[w][0])
                        deltas_pipette_y.append(deltas_pipette[w][1])
                        deltas_pipette_z.append(deltas_pipette[w][2])
                    for h in range(len(deltas_pipette)):
                        deltas_pipette[h]=[sum(deltas_pipette_x[h:len(deltas_pipette_x)]),sum(deltas_pipette_y[h:len(deltas_pipette_y)]),sum(deltas_pipette_z[h:len(deltas_pipette_z)])]
                elim_embryo_new=[]
                for n in range(len(elim_embryo)):
                    elim_embryo_new.append([elim_embryo[n][0]+deltas_pipette[elim_embryo[n][3]][0],elim_embryo[n][1]+deltas_pipette[elim_embryo[n][3]][1],elim_embryo[n][2]+deltas_pipette[elim_embryo[n][3]][2],elim_embryo[n][3],elim_embryo[n][4]])
                injection_results_new(elim_embryo_new,filename,view_1_x,view_1_y,view_2_x,view_2_y,dx_final,dy_final,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,current_x_centroid,current_y_centroid,current_z_centroid,len(positions),q,inj_pressure,pressure_value_current)
    XYZ_Location(20000,20000,8000,current_x_centroid,current_y_centroid,0,ser)
    time.sleep(5)
    print('Press y on video stream')
    # Disconnect xyz stage
    ser.close()
    # Save petri dish and requisite injections  
    mypath=path+'/DSLR_Camera/Row_Col_Petri_Dish'
    path=path+'/DSLR_Camera/'        
    detections_dslr_image(path,filename,mypath,injection_list,x1a_rc_post,y1a_rc_post,x2a_rc_post,y2a_rc_post)
    total_end_time=time.time()
    print('Number of injected embryos = ',injected_embryos)
    print('{} % of dish injected'.format(float(injected_embryos)/float(len(positions))*100))
    print('Average time for injection (s) = ',np.mean(injection_time_list))
    print('Time for injection of dish (min) = ',int((total_end_time-total_start_time)/60)) 
    print('Injection pressure (psi) = ',pressure_value)
    print('Injection pressure time (s) = ',pressure_time)
    print('Injection depth (um) = ',inj_depth)
    print('Injection speed (um/s) = ',inj_speed)

def Main_code_process(p2,__name__act,dish_num,pipette_number,r):
    __name__=__name__act
    if __name__=='__main__':
        r.put([dish_num,pipette_number])
        p2.start()
def Stream_code_process(p1,__name__act):
    __name__=__name__act
    if __name__=='__main__':
        
        p1.start()
