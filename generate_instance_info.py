#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : generate_instance_info_json.py
import os
import xml.etree.ElementTree as ET
import tqdm
import pandas as pd
import math
import json
import heapq
import numpy as np


def p_i(s, n):
    return math.exp((float(s) ** 0.5) / float(n))

def generate_instance_info(args):
    S_Threshold_1 = args.S_Threshold_1
    S_Threshold_2 = args.S_Threshold_2
    w_h_Threshold = args.w_h_Threshold

    pic_path = r"data/input_pic"
    ann_path = r"data/input_xml"

    CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    class_dict = {b: a for a, b in enumerate(CLASSES)}

    xml_list_ = os.listdir(ann_path)

    name_list = []
    class_name = []
    square = []
    w_h = []
    
    for i in tqdm.tqdm(xml_list_):
        ann_i = os.path.join(ann_path, i)
        to_tree = ET.parse(ann_i)
        for element in to_tree.getroot():
            # print(element.tag)
            if element.tag == "object":
                """xmin、xmax、ymin、ymax"""
                xmlbox = element.find('bndbox')
                xmin = xmlbox.find('xmin').text
                ymin = xmlbox.find('ymin').text
                xmax = xmlbox.find('xmax').text
                ymax = xmlbox.find('ymax').text
                class_i = element.find('name').text
                w = int(xmax)
                h = int(ymax)
                s = w * h
                if w < w_h_Threshold and h < w_h_Threshold and S_Threshold_1**2 <= s < S_Threshold_2**2:
                    square.append(s)
                    class_name.append(class_i)
                    name_list.append(i.split('.')[0])
                    w_h.append([w, h])

    print(len(class_name))
    print(len(name_list))
    print(len(square))

    num_result = pd.value_counts(class_name)

    print(num_result)
    num_every_class = [int(num_result[i]) for i in class_name]
    class_id = [class_dict[i] for i in class_name]
    print(len(num_every_class))

    p_all = []
    for s_i, n_i in zip(square, num_every_class):
        p_all.append(p_i(s_i, n_i))

    p_all = list(np.clip(p_all, 1, 180, out=None))


    dict_voc = {}
    ann = {}

    dict_voc['object_pic_path'] = pic_path
    ann['object_name'] = name_list
    ann['p'] = p_all
    ann['name_id'] = class_id
    ann['w_h'] = w_h
    ann['square'] = square
    ann['num_one_class'] = num_every_class
    dict_voc['object_annotation'] = ann

    save_path = 'information/voc_data_v2.json'
    json_str = json.dumps(dict_voc)
    ## save the json file about instances
    with open(save_path, 'w') as json_file:
        json_file.write(json_str)
    ## Output the information of instances
    return name_list,square,num_every_class



