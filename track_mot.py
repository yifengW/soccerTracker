# -*- coding = "utf-8"  -*-

import sys
import os
import numpy as np
import scipy
import pandas as pd
import cv2
from scipy.optimize import linear_sum_assignment as linear_assignment
import glob
import json
from lomo import lomo
from tracker import  Tracklet
import math
from utils import  kalman_filter as kal_ft
from visual import *
import re

class Detection:
    def __init__(self,id,c1_id=-1,c2_id=-1,c1_bbox=None,c2_bbox=None,
                 tx = 0,
                 ty = 0,
                 feature=None):
        self.id = id
        self.c1_id = c1_id
        self.c2_id = c2_id
        self.c1_bbox = c1_bbox
        self.c2_bbox = c2_bbox
        self.bbox = [tx,ty,1,1]
        self.feature = feature



def onlineTrack(lomo_config, idx, notify):
    if idx == 0:
        prefix = "../data"
    else:
        prefix = "../data" + str(idx+1)

    mapping_H1_file = prefix + "/H_first.txt"
    mapping_H2_file = prefix + "/H_second.txt"

    detection_camera1_file = prefix + "/camera1.txt"
    detection_camera2_file = prefix + "/camera2.txt"

    camera1_video_file = prefix + "/first/*.jpg"
    camera2_video_file = prefix + "/second/*.jpg"

    camera1_video = glob.glob(camera1_video_file)
    camera2_video =glob.glob(camera2_video_file)

    #camera1_video = [x.replace('-1','') for x in camera1_video]
    #camera2_video = [x.replace('-2', '') for x in camera2_video]

    camera1_video = sorted(camera1_video,key = lambda x:int(x.replace('-1','').split('/')[-1].split('.jpg')[0]))
    camera2_video = sorted(camera2_video,key = lambda x:int(x.replace('-2', '').split('/')[-1].split('.jpg')[0]))

    mapping_H1 = pd.read_csv(mapping_H1_file,header = None)
    mapping_H2 = pd.read_csv(mapping_H2_file,header = None)
    detection_camera1 = pd.read_csv(detection_camera1_file,header = None)
    detection_camera2 = pd.read_csv(detection_camera2_file,header = None)

    mapping_H1 = mapping_H1.groupby(0)
    mapping_H2 = mapping_H2.groupby(0)
    detection_camera1_group = detection_camera1.groupby(0)
    detection_camera2_group = detection_camera2.groupby(0)

    mapping_H1_key = mapping_H1.indices.keys()
    mapping_H2_key = mapping_H2.indices.keys()
    detection_camera1_key = detection_camera1_group.indices.keys()
    detection_camera2_key = detection_camera2_group.indices.keys()

    assert(len(detection_camera1_key)==len(detection_camera2_key))
    #跟踪器管理
    tracklets = list()
    next_id = 0
    for frame_index  in  detection_camera1_key:
        if notify.is_exit:
            return
        item = frame_index-1

        camera1_detections = np.array(detection_camera1_group.get_group(frame_index).values[:,2:6]).astype(int)
        camera2_detections = np.array(detection_camera2_group.get_group(frame_index).values[:,2:6]).astype(int)

        map_H1_key = list(mapping_H1_key)[item]
        map_H2_key = list(mapping_H2_key)[item]
        H1 = np.array(mapping_H1.get_group(map_H1_key).values[0,1:-1])
        H2 = np.array(mapping_H2.get_group(map_H2_key).values[0,1:-1])

        c1_topview_detections = topViewBoundingbox(camera1_detections,H1)
        c2_topview_detections = topViewBoundingbox(camera2_detections,H2)

        camera1_image = cv2.imread(camera1_video[item])
        camera2_image = cv2.imread(camera2_video[item])

        c1_lomo_features = getLomoFeature(camera1_image,camera1_detections,lomo_config)
        c2_lomo_features = getLomoFeature(camera2_image,camera2_detections,lomo_config)

        c1_detection_num = len(c1_lomo_features)
        c2_detection_num = len(c2_lomo_features)

        appearnce_cost = appearanceCost(c1_lomo_features,c2_lomo_features)
        location_cost = locationCost(c1_topview_detections,c2_topview_detections)

        total_cost = 0.3 * appearnce_cost + 0.7*location_cost

        c1_c2_matches,unmatches_c1,unmatches_c2 = get_detection_matching(total_cost,
                                                                    c1_detection_num,
                                                                    c2_detection_num)

        print("frame_index",frame_index)
        print(c1_c2_matches)
        print(unmatches_c1)
        print(unmatches_c2)

        detections = mergeDetection(c1_c2_matches,unmatches_c1,unmatches_c2,
                                    c1_features=c1_lomo_features,
                                    c2_features=c2_lomo_features,
                                    c1_topview=c1_topview_detections,
                                    c2_topview=c2_topview_detections,
                                    c1_detections=camera1_detections,
                                    c2_detections=camera2_detections)

        for tracklet in tracklets:
            tracklet.predict()

        matches,unmatches_tracklets,unmatches_detections = tracklets_match_detections(tracklets,
                                                                                      detections)
        # print(matches)
        # print(unmatches_tracklets)
        # print(unmatches_detections)
        #匹配住的检测器和跟踪器
        for tracklet,detection in matches:
            tracklet.update(detection)
        #没有匹配住的跟踪器
        for tracklet in unmatches_tracklets:
            tracklet.mark_missed()
        #没有匹配住的检测器
        for  detection in unmatches_detections:
             kt = kal_ft.KalmanFilter()
             mean,covariance = kt.initiate(detection.bbox)
             c1_id = detection.c1_id
             c2_id = detection.c2_id
             tracklets.append(Tracklet(next_id,kt,mean,covariance,c1_id,c2_id))
             next_id+=1


        #更新两个视图的跟踪ID
        new_camera1_tracks = list()
        new_camera2_tracks = list()
        for i,tracklet in enumerate(tracklets):
            #tracklet = tracklets[i]
            if tracklet.is_confirmed():
                if tracklet.c1_id != -1:
                    location =  camera1_detections[tracklet.c1_id,:].tolist()
                    new_camera1_tracks.append([tracklet.id]+location)

                if tracklet.c2_id != -1:
                    location =  camera2_detections[tracklet.c2_id,:].tolist()
                    new_camera2_tracks.append([tracklet.id]+location)


        showView(new_camera1_tracks,camera1_image)
        showView(new_camera2_tracks,camera2_image)

        concatedImage = np.concatenate((camera1_image, camera2_image), axis = 1)

        notify.doRender(cv2.cvtColor(concatedImage,cv2.COLOR_RGBA2RGB))
    #    cv2.imwrite("/Users/bytedance/Demo/dump/camera-{}.jpg".format(frame_index), concatedImage)

        #cv2.namedWindow("camera", 0)
        #cv2.imshow("camera", concatedImage)

        #cv2.waitKey(0)

        # cv2.namedWindow("camera1",0)
        # cv2.namedWindow("camera2",0)
        # cv2.imshow("camera1",camera1_image)
        # cv2.imshow("camera2",camera2_image)
        # cv2.imwrite("./output/mot/c1/camera1-{}.jpg".format(frame_index),camera1_image)
        # cv2.imwrite("./output/mot/c2/camere2-{}.jpg".format(frame_index),camera2_image)
        #cv2.waitKey(0)
        #pass

def cal_distance(tracklet,detections):
    """

    :param tracklets:
    :param detections:
    :return:
    """


    tracklet_x = tracklet.topview_location()[0]
    tracklet_y = tracklet.topview_location()[1]

    distances = list()
    for detection in detections:
        det_x = float(detection.bbox[0])
        det_y = float(detection.bbox[1])
        distances.append(math.sqrt((det_x-tracklet_x)*(det_x-tracklet_x)+(det_y-tracklet_y)*(det_y-tracklet_y)))

    return np.array(distances)


def  tracklets_match_detections(tracklets,detections):
     """

     :param tracklets:
     :param detections:
     :return:
     """
     m = len(tracklets)
     n = len(detections)

     matches = list()
     unmatches_tracklets = list()
     unmatches_detections = list()

     if m ==0 or n == 0:
         return [],tracklets,detections

     cost = np.zeros((m,n))

     for row,tracklet in enumerate(tracklets):
         cost[row,:] = cal_distance(tracklet,detections)

     mean_cost = np.mean(cost)
     std_cost = np.std(cost)

     cost = (cost-mean_cost)/std_cost

     indices = linear_assignment(cost)

     for row,tracklet in enumerate(tracklets):
         if row not in indices[0]:
             unmatches_tracklets.append(tracklet)

     for col,detection in enumerate(detections):
         if col not in indices[1]:
             unmatches_detections.append(detection)


     rows = indices[0]
     cols = indices[1]

     for row,col in zip(rows, cols):
         if cost[row,col]> -0.5:
             unmatches_tracklets.append(tracklets[row])
             unmatches_detections.append(detections[col])
             continue

         matches.append([tracklets[row],detections[col]])

     return matches, unmatches_tracklets,unmatches_detections


def mergeDetection(mathes,unmatches_c1,unmatches_c2,
                   c1_features,
                   c2_features,
                   c1_topview,
                   c2_topview,
                   c1_detections,
                   c2_detections):
    id = 0
    detections = list()
    c1_features = np.array(c1_features)
    c2_features = np.array(c2_features)
    c1_topview = np.array(c1_topview)
    c2_topview = np.array(c2_topview)
    #整合匹配的
    for c1_id,c2_id in mathes:
        feature = (c1_features[c1_id,:]+c2_features[c2_id,:])/2
        bbox = (c1_topview[c1_id,:]+c2_topview[c2_id,:])/2
        c1_bbox = c1_detections[c1_id,:]
        c2_bbox = c2_detections[c2_id,:]
        detections.append(Detection(id,c1_id=c1_id,c2_id=c2_id,
                                    c1_bbox=c1_bbox,
                                    c2_bbox=c2_bbox,
                                    tx=bbox[0],
                                    ty=bbox[1],
                                    feature =feature))
        id += 1

    for c1_id in unmatches_c1:
        feature = c1_features[c1_id,:]
        bbox = c1_topview[c1_id,:]
        c1_bbox = c1_detections[c1_id,:]
        detections.append(Detection(id,c1_id = c1_id,
                                    c2_id = -1,
                                    c1_bbox = c1_bbox,
                                    tx = bbox[0],
                                    ty = bbox[1],
                                    feature = feature))
        id += 1

    for c2_id in unmatches_c2:
        feature = c2_features[c2_id,:]
        bbox = c2_topview[c2_id,:]
        c2_bbox = c2_detections[c2_id,:]
        detections.append(Detection(id,c1_id = -1,
                                    c2_id = c2_id,
                                    c2_bbox = c2_bbox,
                                    tx = bbox[0],
                                    ty = bbox[1],
                                    feature = feature))
        id += 1

    return detections

def topViewBoundingbox(c_detections,H):
    """

    :param c_detections:
    :param H:
    :return:
    """
    assert(H.shape[0]==9)
    topview_detections = list()
    H = H.reshape((3,3)).astype(float)
    m = c_detections.shape[0]
    one = np.ones((m,1))
    c_detections = np.concatenate([c_detections,one],axis=1)

    for i  in range(m):
        coordinate_tl = np.array([[[c_detections[i,0],c_detections[i,1]]]])
        coordinate_br = np.array([[[c_detections[i,0]+c_detections[i,2],
                                    c_detections[i,1]+ c_detections[i,3]]]])
        new_coordinate_tl = cv2.perspectiveTransform(coordinate_tl,H).squeeze()
        new_coordinate_br = cv2.perspectiveTransform(coordinate_br,H).squeeze()
        topview_detections.append(new_coordinate_tl.tolist()+
                                   new_coordinate_br.tolist())

    return topview_detections

def get_detection_matching(cost,c1_num,c2_num):
    """

    :param appearnce_cost:
    :return:
    """
    indices = linear_assignment(cost)
    matches = list()
    unmatches_c1 = list()
    unmatches_c2 = list()

    for  row in range(c1_num):
        if row not in indices[0]:
            unmatches_c1.append(row)

    for  col in range(c2_num):
        if col not in indices[1]:
            unmatches_c2.append(col)

    rows = indices[0]
    cols = indices[1]

    for row,col in zip(rows, cols):
        if cost[row,col] > -0.5:
            unmatches_c1.append(row)
            unmatches_c2.append(col)
            continue
        matches.append([row,col])


    return matches,unmatches_c1,unmatches_c2

def appearanceCost(c1_lomo_features,c2_lomo_features):
    """

    :param c1_lomo_features:
    :param c2_lomo_features:
    :return:
    """
    m = len(c1_lomo_features)
    n = len(c2_lomo_features)

    c1_features = np.array(c1_lomo_features) #m*f
    c2_features = np.array(c2_lomo_features) #n*f

    c1_features = np.transpose(np.tile(c1_features,(n,1,1)),(1,0,2))
    c2_features = np.tile(c2_features,(m,1,1))

    cost =np.sum(np.power(c1_features-c2_features,2),axis=2)

    mean_cost = np.mean(cost)
    std_cost = np.std(cost)

    cost = (cost-mean_cost)/(std_cost+1e-10)

    return cost


def locationCost(c1_topview_detections, c2_topview_detections):
    """

    :param c1_topview_detections:
    :param c2_topview_detections:
    :return:
    """
    # c1_topview_detections =
    # c2_topview_detections =

    c1_topview = np.array(c1_topview_detections)[:,2:].copy()
    c2_topview = np.array(c2_topview_detections)[:,2:].copy()

    # print(id(c1_topview))
    # print(id(c1_topview_detections))
    #
    # # c1_topview_detections = np.array(c1_topview_detections)
    # # c2_topview_detections = np.array(c2_topview_detections)
    # #
    # # c1_topview = c1_topview_detections[:,2:]

    m = c1_topview.shape[0]
    n = c2_topview.shape[0]

    c1_topview = np.transpose(np.tile(c1_topview,(n,1,1)),(1,0,2))
    c2_topview = np.tile(c2_topview,(m,1,1))

    cost = np.sqrt(np.sum(np.power(c1_topview-c2_topview,2),axis=2))

    mean_cost = np.mean(cost)
    std_cost = np.std(cost)

    cost = (cost-mean_cost)/(std_cost+1e-10)

    return  cost


def getLomoFeature(image,detections,lomo_config):
    """
    :param image: numpy (h,w,3)
    :param detections:  (num,4)
    :return:
    """
    m = detections.shape[0]
    lomo_features = list()
    for i  in range(m):
        tf_x = detections[i,0]
        tf_y = detections[i,1]
        br_x = detections[i,0]+detections[i,2]
        br_y = detections[i,1]+detections[i,3]
        img = image[tf_y:br_y,tf_x:br_x,:]
        img = cv2.resize(img,(128,64))
        lomo_feature = lomo.LOMO(img,lomo_config)
        lomo_features.append(lomo_feature)

    return lomo_features





