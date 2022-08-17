# -*- coding = "utf-8"

import cv2
import numpy as np

colors = [(255,0.255),(128,255,128),(255,0,128),(255,128,128),(0,255,255),(255,0,0),
(255,128,128),(180,180,180)]

def showImage(match,unmatch_camera1,unmatch_camera2,camera1_points,
              camera2_points,
              camera1_index,
              camera2_index):
    """

    :param match:
    :param unmatch_camera1:
    :param unmatch_camera2:
    :param camera1_point:
    :param camera2_point:
    :return:
    """
    w = 1920
    h = 1080
    img = np.zeros(shape=(h,w,3))
    track_id = 0
    cv2.namedWindow("play_ground",flags=0)
    for camera1_idx,camera2_idx in match:
        index1 = camera1_index.tolist().index(camera1_idx)
        #index2 = camera2_index.find(camera2_idx)
        point = camera1_points[index1]*15
        #camera2_point = camera2_points[index2]
        cv2.circle(img,center=(int(point[1]),int(point[2])),radius=5,color=(255,0,0),thickness=3)

        cv2.putText(img,str(track_id),(int(point[1]),int(point[2])),cv2.FONT_HERSHEY_COMPLEX,2,
                    (255,255,255),2)
        track_id +=1

    for camera1_idx in unmatch_camera1:
        index1 = camera1_index.tolist().index(camera1_idx)
        point = camera1_points[index1].astype(int)*15
        cv2.circle(img,center=(point[1],point[2]),radius=5,color=(0,255,0),thickness=3)
        cv2.putText(img,str(track_id),(point[1],point[2]),cv2.FONT_HERSHEY_COMPLEX,2,
                    (255,255,255),2)
        track_id+=1

    for camera2_idx in unmatch_camera2:
        index2 = camera2_index.tolist().index(camera2_idx)
        point = camera2_points[index2].astype(int)*15
        cv2.circle(img,center=(point[1],point[2]),radius=5,color=(0,0,255),thickness=3)
        cv2.putText(img,str(track_id),(point[1],point[2]),cv2.FONT_HERSHEY_COMPLEX,2,
                    (255,255,255),2)
        track_id+=1

    cv2.imshow("play_ground",img)
    cv2.waitKey(0)



def showView(tracks,img):
    for track in tracks:
        location = track[1:]
        tx = int(location[0])
        ty = int(location[1])
        bx = int(location[0]+location[2])
        by = int(location[1]+location[3])

        id = track[0]
        color = colors[(id+3)%len(colors)]
        cv2.rectangle(img,pt1=(tx,ty),pt2=(bx,by),color=color,thickness=5)
        cv2.putText(img,str(id),org=(tx+5,ty+5),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=2,
                    color=(0,0,255),thickness=3)
    pass


def show(tracklets,detections):
    """

    :param tracklets:
    :param detections:
    :return:
    """
    w = 1920
    h = 1080
    black_image = np.zeros(shape = (h,w,3))

    w_rate = w/115
    h_rate = h/74
    tracklet_color = (255,0,0)
    detection_color = (0,255,255)
    #cv2.namedWindow("play_ground",flags=0)
    # for tracklet in tracklets:
    #     if tracklet.is_confirmed:
    #         x = int(tracklet.topview_location()[0]*w_rate)
    #         y = int(tracklet.topview_location()[1]*h_rate)
    #         cv2.circle(black_image,center=(x,y),radius=5,color=tracklet_color,thickness=3)
    #         cv2.putText(black_image,str(tracklet.id),(x,y),cv2.FONT_HERSHEY_COMPLEX,2,
    #                     (0,0,255),1)

    for detection  in detections:
        x = int(detection.bbox[0]*w_rate)
        y = int(detection.bbox[1]*h_rate)
        cv2.circle(black_image,center=(x,y),radius=5,color=detection_color,thickness=3)
        cv2.putText(black_image,str(detection.id),(x,y),cv2.FONT_HERSHEY_COMPLEX,2,
                    (255,0,255),1)


    #cv2.imshow("play_ground",black_image)
    return black_image


def save_output_txt(tracklets,frame):
    """

    :param tracklets:
    :param frame:
    :return:
    """
    outputs = list()
    for track in tracklets:
        location = track[1:]
        tx = int(location[0])
        ty = int(location[1])
        w =  int (location[2])
        h = int(location[3])
        id = track[0]

        output = np.array([frame,id,tx,ty,w,h,-1,-1,-1,-1])
        outputs.append(output)

    return np.array(outputs)

