# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:38:53 2017

@author: Rasmus
"""

import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import os
import glob
from random import shuffle
from shutil import copyfile



def split_data_set(data_dir, new_root):
    # old dirs
    image_dir = os.path.join(data_dir,'images')    
    label_dir = os.path.join(data_dir,'labels')
    # new dirs
    train_im_dir = os.path.join(new_root,'training','images')    
    train_label_dir = os.path.join(new_root,'training','labels')
    val_im_dir = os.path.join(new_root,'validation','images')    
    val_label_dir = os.path.join(new_root,'validation','labels')
    test_im_dir = os.path.join(new_root,'testing','images')    
    test_label_dir = os.path.join(new_root,'testing','labels')
    # create new dirs
    os.makedirs(train_im_dir)
    os.makedirs(train_label_dir)
    os.makedirs(val_im_dir)
    os.makedirs(val_label_dir)
    os.makedirs(test_im_dir)
    os.makedirs(test_label_dir)

    os.chdir(image_dir)
    orig_im_files = [x for x in glob.glob("*.png")]
    shuffle(orig_im_files)
    nbr_files = len(orig_im_files)
    for i, image_file in enumerate(orig_im_files):
        print('#######    ' + image_file + '    #######')
        file_no_ext = os.path.splitext(image_file)[0]
        im_path = os.path.join(image_dir, image_file)
        txt_path = os.path.join(label_dir, file_no_ext) + '.txt'   
        if (i < 0.6*nbr_files): # training
            new_im_path = os.path.join(train_im_dir, image_file)
            new_txt_path = os.path.join(train_label_dir, file_no_ext) + '.txt'
        elif i < 0.8*nbr_files: # validation 
            new_im_path = os.path.join(val_im_dir, image_file)
            new_txt_path = os.path.join(val_label_dir, file_no_ext) + '.txt'
        else:
            new_im_path = os.path.join(test_im_dir, image_file)
            new_txt_path = os.path.join(test_label_dir, file_no_ext) + '.txt'
        
        copyfile(im_path, new_im_path)
        copyfile(txt_path, new_txt_path)


def bbox_from_file(path):
    with open(path, 'r') as f:
        data_str = f.read()
        bbox_lst = [float(x) for x in str.split(data_str)[4:8]]
    return np.array(bbox_lst)

def line_to_bbox(line_str):
    bbox_lst = [float(x) for x in str.split(line_str)[4:8]]
    return np.array(bbox_lst)


def bboxes_from_file(path):
    bboxes = []
    with open(path, 'r') as file:
        for row in file:
            data_lst = str.split(row)
            occluded = float(data_lst[2])
            print(occluded)
            if (data_lst[0] == 'Pedestrian') and occluded == 0:
                bboxes.append([float(x) for x in data_lst[4:8]])
    return bboxes


def area(bbox):
    return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])


def calculate_overlap(bbox1, bbox2):   
    x1 = max(bbox1[0],bbox2[0])    
    y1 = max(bbox1[1],bbox2[1])
    x2 = min(bbox1[2],bbox2[2])    
    y2 = min(bbox1[3],bbox2[3]) 
    
    if x2<x1 or y2<y1:
        return 0
    else:
        return (x2-x1)*(y2-y1)

def draw_overlap(bbox1, bbox2, image):
    x1 = int(round(max(bbox1[0],bbox2[0])))  
    y1 = int(round(max(bbox1[1],bbox2[1])))
    x2 = int(round(min(bbox1[2],bbox2[2])))  
    y2 = int(round(min(bbox1[3],bbox2[3])))
    cv2.rectangle(image, (x1,y1), (x2,y2), (255, 0, 0), 2)
    
def compare_boxes(bbox1, bbox2, threshold):
    overlap = (calculate_overlap(bbox1, bbox2))
    area1 = area(bbox1)
    area2 = area(bbox2)
    overlap_fraction = overlap/max(area1,area2)
    return (overlap_fraction > threshold)
        
def overlay_bbox(image, bbox, color):
    bottom_left = (int(round(bbox[0])), int(round(bbox[1]))   ) 
    top_right = (int(round(bbox[2])), int(round(bbox[3])))
    cv2.rectangle(image, bottom_left, top_right, color, 2)

def overlay_bboxes(image, bboxes, color):
    for i_bbox in bboxes:
        overlay_bbox(image, i_bbox, color)
    
    

def evaluate(detected_bboxes, control_bboxes, threshold):
    nbr_detected = len(detected_bboxes)
    was_matched = [False for x in range(nbr_detected)]
    hits = 0
    for i_control in control_bboxes:
        for i, i_detected in enumerate(detected_bboxes):
            if (not was_matched[i]) and compare_boxes(i_control, i_detected, threshold):
                print(i)
                hits += 1
                was_matched[i] = True
    false_positives = nbr_detected -sum(was_matched)
    return hits, false_positives

def test_hog(hog, data_dir, threshold=0.5):
    image_dir = os.path.join(data_dir,'images')    
    label_dir = os.path.join(data_dir,'labels')
    os.chdir(image_dir)

    tot_nbr_control_bboxes = 0
    tot_nbr_detected_boxes = 0
    total_true_positives = 0
    total_false_positives = 0
    
    

    for image_file in glob.glob("*.png"):
        print('#######    ' + image_file + '    #######')
        file_no_etx = os.path.splitext(image_file)[0]
        txt_path = os.path.join(label_dir, file_no_etx) + '.txt'
        image_path = os.path.join(image_dir, image_file)
        control_bboxes = bboxes_from_file(txt_path)
        
        image = cv2.imread(image_path)
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), 
         padding=(16, 16), scale=1.05)
        all_bboxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        best_bboxes = non_max_suppression(all_bboxes, overlapThresh = 0.5, probs = None)

        hits, false_positives = evaluate(best_bboxes, control_bboxes, threshold)
        print('Hits:' + str(hits))
        print ('False positives: ' + str(false_positives))
        tot_nbr_control_bboxes += len(control_bboxes)
        tot_nbr_detected_boxes += len(best_bboxes)
        total_true_positives += hits
        total_false_positives += false_positives
        try:
            print('Temporary sensitivity: ' + str(total_true_positives/tot_nbr_control_bboxes))
            print('Temporary false pos. rate: ' + str(total_false_positives/tot_nbr_detected_boxes))
        except:
            print('---')
        # viz
        #overlay_bboxes(image, control_bboxes, (0, 255, 0))
        #overlay_bboxes(image, all_bboxes, (255, 0, 0))
        #overlay_bboxes(image, best_bboxes, (0, 0, 255))
        #cv2.imshow('', image)
        #cv2.waitKey(0)
        
    sensitivity = total_true_positives/tot_nbr_control_bboxes
    false_positive_rate = total_false_positives/tot_nbr_detected_boxes
    return sensitivity, false_positive_rate

data_dir = 'C:\\Users\\Rasmus\\Documents\\GitHub\\pedestrian-identification'
#data_dir = 'D:\\kitti-data\\testing'
hog = cv2.HOGDescriptor("hog.xml")
print(test_hog(hog,data_dir))



"""
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
hog.save("hog.xml")

image = cv2.imread(image_path)

(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)

x1 = rects[0][0]
y1 = rects[0][1]
x2 = x1 + rects[0][2]
y2 = y1 + rects[0][3]
bbox2 = [x1, y1, x2, y2]

overlap = (calculate_overlap(bbox1, bbox2))
print(overlap)
area1 = area(bbox1)
overlap_fraction = overlap/area1
print(overlap_fraction)
area_fraction = area1/area(bbox2)
area_error = abs(area_fraction-1)
print(area_error)
"""
