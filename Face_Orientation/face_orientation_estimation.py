## USAGE
## py face_orientation_estimation.py --weights weights/face.pt

# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import copy
import math 

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import os

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xywh, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    #cv2.rectangle(img, (x1,y1), (x2, y2), (255,0,255), thickness=5, lineType=cv2.LINE_AA)	
	
    clors = [(0,255,255),(0,255,255),(0,255,255),(0,255,255),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        #cv2.circle(img, (point_x, point_y), 4, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    #cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect_one(model, image_path, device, rotation):
    # Load model
    img_size = 800
    conf_thres = 0.02  # 0.3 in og
    iou_thres = 0.5   # 0.5 in og


	
	### FACE ORIENTATION ESTIMATION
	### Compute scores from all orientations at 0,90,180,270 degrees
	
	# 0 degree counter-clockwise rotation
    if rotation == 0:
        orgimg = cv2.imread(image_path)  # BGR
        img0 = copy.deepcopy(orgimg)
	# 90 degree counter-clockwise rotation
    elif rotation == 90:
        img = cv2.imread(image_path)   
        img_rgb = img.copy()
        img_rot = cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

        orgimg = img_rot.copy() # BGR
        img0 = copy.deepcopy(orgimg)
    # 180 degree counter-clockwise rotation
    elif rotation == 180:
        img = cv2.imread(image_path)   
        img_rgb = img.copy()
        img_rot = cv2.rotate(img_rgb, cv2.ROTATE_180)

        orgimg = img_rot.copy() # BGR
        img0 = copy.deepcopy(orgimg)
    # 270 degree counter-clockwise rotation
    elif rotation == 270:
        img = cv2.imread(image_path)   
        img_rgb = img.copy()
        img_rot = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)

        orgimg = img_rot.copy() # BGR
        img0 = copy.deepcopy(orgimg)		
	

	
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    #print('img.shape: ', img.shape)
    #print('orgimg.shape: ', orgimg.shape)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()
            
            conf_vec = []
            xywh_vec = []
            landmarks_vec = []
            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                xywh_vec.append(xywh)
                conf = det[j, 4].cpu().numpy()
                conf_vec.append(conf)
                landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                landmarks_vec.append(landmarks)
                class_num = det[j, 15].cpu().numpy()
                orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
                
				
    x = file.split("/")
    name = x[-1]
	
	### SAVE PREDICTIONS
    detections = []
    # If no detection
    if det.nelement()==0:
        detections = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    else:
        # Check number of detections
        numDet = round(det.nelement() / 16)

        for d in range(numDet):
            xywh = xywh_vec[d]
            h,w,c = orgimg.shape
            x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
            y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
            x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
            y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)

            xmin = x1
            ymin = y1
            width = x2-x1
            height = y2-y1
			
            landmarks = landmarks_vec[d]
            land = []
            for i in range(5):
                point_x = int(landmarks[2 * i] * w)
                land.append(point_x)
                point_y = int(landmarks[2 * i + 1] * h)	
                land.append(point_y)

            detections.append(xmin)
            detections.append(ymin)
            detections.append(width)
            detections.append(height)
            detections.append(conf_vec[d])
            detections.append(land[0])
            detections.append(land[1])
            detections.append(land[2])
            detections.append(land[3])
            detections.append(land[4])
            detections.append(land[5])
            detections.append(land[6])
            detections.append(land[7])
            detections.append(land[8])
            detections.append(land[9])
 	
    if not os.path.exists('north_oriented_faces'):
        os.makedirs('north_oriented_faces')
    fullname_dir = 'north_oriented_faces/' + name
    cv2.imwrite(fullname_dir, orgimg)
   
    detected_faces = np.array(detections)
	
	
	### FACE ORIENTATION ESTIMATION
	### Compute NELA with predicted landmarks
    if det.nelement()==0:
        angle = []
    else:
	    ## Calculate projected point
        p0 = [detections[5],detections[6]] # X left eye, Y left eye
        p1 = [detections[7],detected_faces[8]] # X right eye, Y right eye
        q = [detections[9],detections[10]] # X nose, Y nose
        a = np.array([[-q[0]*(p1[0]-p0[0]) - q[1]*(p1[1]-p0[1])],[ -p0[1]*(p1[0]-p0[0]) + p0[0]*(p1[1]-p0[1])]])
        b = np.array([[p1[0] - p0[0], p1[1] - p0[1]], [p0[1] - p1[1], p1[0] - p0[0]]])
        #print("p0, p1, q, a, b, ProjPoint, proj_point:", p0, p1, q, a, b)
		
		# check if eye_line or nose_line == 0
        if np.all(a == 0) or np.all(b == 0):
            angle = []
            return detected_faces, angle
        else:
            ProjPoint = -1 * np.matmul(np.linalg.inv(b),a)
            proj_point = np.round(ProjPoint.transpose())
			
			## Calculate angle (NELA)
			#proj_point_list = np.tolist(proj_point)
            u = np.array([proj_point[0][0],proj_point[0][1]]) - np.array(q)
            v = np.array([w,detections[10]])- np.array(q)
            val = np.dot(u,v) / np.linalg.norm(u) / np.linalg.norm(v)
            Theta = np.arccos(np.amin(np.array([np.amax(val,-1),1])))
            ThetaInDegrees = np.rad2deg(Theta)
            vc = np.cross(np.append(u,0), np.append(v,0))
			
            anticlock = vc[2] > 0
            if anticlock:
                angle = ThetaInDegrees
            else:
                angle = 360 - ThetaInDegrees
    
    return detected_faces, angle



if __name__ == '__main__':
    directory = 'data/CHEO_uncovered_pt_threshold_5/'
    #directory = 'data/imgs/'

    t = time.time()

    textfile = open("predictions_face_orientation_uncov.txt", "w")
	
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(file):
            parser = argparse.ArgumentParser()
            # parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
            parser.add_argument('--weights', nargs='+', type=str, default='face.pt', help='model.pt path(s)')
            parser.add_argument('--image', type=str, default=file, help='source')  # file/folder, 0 for webcam
            parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
            parser.add_argument('--rotation', type=int, default=90, help='rotation angle')
            opt = parser.parse_args()
            #print(opt)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model(opt.weights, device)
			
			### FACE ORIENTATION ESTIMATION
			### Extract SCORES and NELA from all orientations
            detected_faces_0, angle_0 = detect_one(model, opt.image, device, 0)
            detected_faces_90, angle_90 = detect_one(model, opt.image, device, 90)			
            detected_faces_180, angle_180 = detect_one(model, opt.image, device, 180)
            detected_faces_270, angle_270 = detect_one(model, opt.image, device, 270)
			
			
            SCORES = []
            SCORES.append(detected_faces_0[4])
            SCORES.append(detected_faces_90[4])
            SCORES.append(detected_faces_180[4])
            SCORES.append(detected_faces_270[4])	
            saved_scores = list.copy(SCORES)
            print("SCORES :", SCORES)
			
            NELA = []
            NELA.append(angle_0)
            NELA.append(angle_90)
            NELA.append(angle_180)
            NELA.append(angle_270)			
            print("NELA :", NELA)

			
            notFound = True
			
			# If no detection in any orientation
            if all(elem == 0 for elem in SCORES):
                detected_faces_0, angle_0 = detect_one(model, opt.image, device, 0)
                detected_faces = detected_faces_0
                print("no orientation found")
                selected_NELA = 'noNELA'
                notFound = False
					
			# If at least one detection	among orientations
            while notFound:
                max_score = max(SCORES)
                idx = SCORES.index(max_score) 
                angle = NELA[idx]
                if isinstance(angle,list) or math.isnan(angle):
                    SCORES[idx] = -1
                elif (angle > 45) and (angle <= 135):
                    north_pred = idx
                    print("north_pred: ",north_pred)
                    selected_NELA = str(NELA[idx])
					
                    #Save identified detection
                    if north_pred == 0:
                        detected_faces_0, angle_0 = detect_one(model, opt.image, device, 0)
                        detected_faces = detected_faces_0
                        rotation = 0
                    elif north_pred == 1:
                        detected_faces_90, angle_90 = detect_one(model, opt.image, device, 90)
                        detected_faces = detected_faces_90	
                        rotation = 90						
                    elif north_pred == 2:
                        detected_faces_180, angle_180 = detect_one(model, opt.image, device, 180)	
                        detected_faces = detected_faces_180	
                        rotation = 180						
                    elif north_pred == 3:
                        detected_faces_270, angle_270 = detect_one(model, opt.image, device, 270)			
                        detected_faces = detected_faces_270
                        rotation = 270
						
                    notFound = False
                else:
                    SCORES[idx] = -1
                    #print("new scores: ",SCORES)
            	
                if all(elem == -1 for elem in SCORES):
                    detected_faces_0, angle_0 = detect_one(model, opt.image, device, 0)
                    detected_faces = detected_faces_0
                    print("no orientation found")
                    rotation = 0
                    selected_NELA = 'noNELA'
                    notFound = False

			
			### Save all predictions
            detected = np.array_str(detected_faces)
            detected1 = " ".join(detected.split())
            detected2 = detected1.lstrip()
            yolo_results = detected2.replace("[ ", "").replace("]", "").replace("[","")
			
            or_scores = ['%.5f' % e for e in saved_scores]
            face_orient_scores = " ".join(str(e)for e in or_scores)
            
            face_orient_nela = ''
            for e in NELA:
                if isinstance(e,list) or math.isnan(e):
                    face_orient_nela = face_orient_nela + ' noPred'
                else:
                    face_orient_nela = face_orient_nela + ' %.2f' % e
            face_orient_nela = face_orient_nela.lstrip()
			
            textfile.write(filename +  "\n")
            textfile.write(yolo_results +  "\n")
            textfile.write(face_orient_scores + "\n")
            textfile.write(face_orient_nela + "\n")
            textfile.write(selected_NELA + "\n")
            textfile.write(str(rotation) + "\n")
			
  	
    elapsed = time.time() - t
    print('Elapsed detection time for dataset: ', elapsed)	
    textfile.close()
			
	
