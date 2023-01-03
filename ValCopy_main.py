#!/usr/bin/python
# -*- coding: utf-8 -*-
# @File    : generateDataset.py
# @Software: PyCharm
import os
import shutil
import numpy as np
import random
import math
from PIL import Image
import xml.etree.ElementTree as ET

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import tqdm
import datetime
import json
import argparse
from generate_instance_info import generate_instance_info

def copy_allfiles(src, dest):

    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)


def find_w_h(bbox, s_w, s_h):  ## bbox:x1,x2,y1,y2,s_w,s_h is the width and height of subfigure
    if s_w >= 64:
        w = bbox[0]
    else:
        w = random.randint(bbox[0], bbox[1] - s_w)

    if s_h >= 64:
        h = bbox[2]
    else:
        h = random.randint(bbox[2], bbox[3] - s_h)
    return w, h  ## Upper left coordinate


def p_i(list_for_s, list_for_num, i):
    return math.exp(float(list_for_s[i]) ** 0.5) / float(list_for_num[i])


def alias_setup(probs):

    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=int)
    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob  # probability
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.


    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):

    K = len(J)
    # Draw from the overall uniform mixture.
    kk = int(np.floor(np.random.rand() * K))
    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def IOU(box1, box2):
    """
    :param box1:[x1,y1,x2,y2]
    :param box2:[x1,y1,x2,y2]
    :return: iou_ratio
    """

    width1 = abs(box1[2] - box1[0])
    height1 = abs(box1[1] - box1[3])
    width2 = abs(box2[2] - box2[0])
    height2 = abs(box2[1] - box2[3])
    x_max = max(box1[0], box1[2], box2[0], box2[2])
    y_max = max(box1[1], box1[3], box2[1], box2[3])
    x_min = min(box1[0], box1[2], box2[0], box2[2])
    y_min = min(box1[1], box1[3], box2[1], box2[3])
    iou_width = x_min + width1 + width2 - x_max
    iou_height = y_min + height1 + height2 - y_max
    if iou_width <= 0 or iou_height <= 0:
        iou_ratio = 0
    else:
        iou_area = iou_width * iou_height
        box1_area = width1 * height1
        box2_area = width2 * height2
        iou_ratio = iou_area / (box1_area + box2_area - iou_area)
    return iou_ratio


def find_child(node, with_name):

    for element in list(node):
        if element.tag == with_name:
            return element
            # all_element.append(element)
        elif list(element):
            sub_result = find_child(element, with_name)
            if sub_result is not None:
                return sub_result
    # return all_element
    return None


def get_node_parent_info(tree, node):
    """
    Return tuple of (parent, index) where:
        parent = node's parent within tree
        index = index of node under parent
    """
    parent_map = {c: p for p in tree.iter() for c in p}
    parent = parent_map[node]
    return parent, list(parent).index(node)


def replace_node(to_tree, node_name, node, new_xy, w,
                 h):  ##new_xy is the new upper left coordiante, w, h is the width and height of subfigure

    """
    Replace node with given node_name in to_tree with
    the same-named node from the from_tree
    """
    # Find nodes of given name ('car' in the example) in each tree

    for element in list(node):
        if element.tag == node_name:
            """修改xmin、xmax、ymin、ymax"""
            xmlbox = element.find('bndbox')
            xmin = xmlbox.find('xmin').text
            xmax = xmlbox.find('xmax').text
            ymin = xmlbox.find('ymin').text
            ymax = xmlbox.find('ymax').text
            # print(xmin,xmax,ymin,ymax)

            xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
            new_xmin = str(int(new_xy[0]))  ## x1
            new_xmax = str(int(new_xy[0]) + int(w))  ## x2
            new_ymin = str(int(new_xy[1]))  ## y1
            new_ymax = str(int(new_xy[1]) + int(h))  ## y2
            xmlbox.find('xmin').text = new_xmin
            xmlbox.find('xmax').text = new_xmax
            xmlbox.find('ymin').text = new_ymin
            xmlbox.find('ymax').text = new_ymax
            # print("new_xmin", new_xmin)
            # print("new_xmax", new_xmax)
            # print("new_ymin", new_ymin)
            # print("new_ymax", new_ymax)
            # print("________")
            from_node = element

            # from_node = find_child(from_tree.getroot(), node_name)
            to_node = find_child(to_tree.getroot(), node_name)

            # Find where to substitute the from_node into the to_tree

            to_parent, to_index = get_node_parent_info(to_tree, to_node)

            # Replace to_node with from_node
            # to_parent.remove(to_node)

            # for i in from_node:
            to_parent.insert(to_index, from_node)


def _to_list(path):

    data1 = []
    data2 = []
    data0 = []

    for line in open(path, "r"):
        # print(line)
        data0.append(line[:-1])
        line = line[:-1] + ".jpg"
        data1.append(line)
    for line in open(path, "r"):
        # print(line)
        line = line[:-1] + ".xml"
        data2.append(line)
    return data0, data1, data2


def ValCopy(args):

    InstanceImg_path = args.InstanceImg_Path  ## path for instances
    InstanceXml_path = args.InstanceXml_Path  ## path of the xml files of instances
    N = args.N

    AugImg_path = os.path.join(args.AugImg_path, 'map_pic26-64-90-90-{}'.format(N))
    AugXml_path = os.path.join(args.AugXml_path, 'map_xml26-64-90-90-{}'.format(N))

    OriImg_path = args.OriImg_path


    probs_file = "save_txt/probs/probs.txt"
    details_file = "save_txt/details/details.txt"


    all_bbox = np.load('all_bbox-len-90-var-90.npy', allow_pickle=True)
    all_bbox = all_bbox.tolist()
    # print(len(all_bbox))

    list_for_px = []
    for id in range(len(list_for_num)):
        list_for_px.append(p_i(list_for_s, list_for_num, id))
    probs = [i / sum(list_for_px) for i in list_for_px]
    print(max(probs), min(probs), len(probs), sum(probs))
    with open(probs_file, "a") as f:
        for M_image, prob in zip(list_for_select_pic, probs):
            xx = '{}{}{}'.format(str(M_image), '\t', str(prob))
            f.write(xx + "\n")


    if os.path.exists(path=AugImg_path):
        shutil.rmtree(AugImg_path)
    os.mkdir(AugImg_path)

    if os.path.exists(AugXml_path):
        shutil.rmtree(AugXml_path)
    os.mkdir(AugXml_path)


    src = os.path.join(OriImg_path, 'ImageSets/Main/train.txt')
    TrainInfo_path = 'information/train.txt'
    if os.path.exists(TrainInfo_path):
        os.remove(TrainInfo_path)
    shutil.copyfile(src, TrainInfo_path)

    data0, list_for_train_pic, list_for_train_xml = _to_list(src)
    # ============================================================================
    path_for_train_img = os.path.join(OriImg_path, 'JPEGImages')
    # list_for_train_pic = os.listdir(train_pic_path)
    path_for_train_xml = os.path.join(OriImg_path, 'Annotations')
    # ============================================================================


    num_for_all = 0
    num_im = 0

    list_for_NN = []

    for idx_0, pic in enumerate(tqdm.tqdm(list_for_train_pic)):


        # list_for_s_i = []
        NN = min(N, len(all_bbox[idx_0]))
        list_for_NN.append(NN)
        if NN == 0:
            continue

        num_for_all += 1
        # print(pic)

        J, q = alias_setup(probs)
        X = np.zeros(NN)
        for nn in range(NN):
            X[nn] = alias_draw(J, q)
        son_img_file = []
        for idx in X:
            son_img_file.append(list_for_select_pic[int(idx)])

        with open(details_file, "a") as f:
            xx = '{}{}{}'.format(str(pic), '\t', str(son_img_file))
            f.write(xx + "\n")
        # ===============================================================

        M_Img = Image.open(os.path.join(path_for_train_img, pic))
        M_Img = M_Img.convert("RGBA")
        # M_Img_w, M_Img_h = M_Img.size
        to_tree = ET.parse(os.path.join(path_for_train_xml, pic[:-4] + ".xml"))
        # print("to tree",to_tree)


        _bbox = random.sample(all_bbox[idx_0], NN)

        for idx_1, i in enumerate(son_img_file):
            S_Img = Image.open(InstanceImg_path + "/" + i + ".jpg").convert(
                "RGBA")


            S_Img_w, S_Img_h = S_Img.size
            from_tree = ET.ElementTree(file=InstanceXml_path + "//" + i + ".xml")


            __bbox = _bbox[idx_1]
            w, h = find_w_h(__bbox, S_Img_w, S_Img_h)
            # coordinate = (h, w)
            coordinate = (w, h)


            M_Img.paste(S_Img, coordinate, S_Img)
            replace_node(to_tree, 'object', from_tree.getroot(), new_xy=coordinate, w=S_Img_w,
                         h=S_Img_h)

        # print(pic, 'xml is finished')

        M_Img.save(
            os.path.join(AugImg_path, pic[:-4] + "_0" + ".png"))
        to_tree.write(os.path.join(AugXml_path, pic[:-4] + "_0" + ".xml"))


    print("name updating")
    li = os.listdir(AugImg_path)
    for filename in li:
        newname = filename
        newname = newname.split(".")
        if newname[-1] == "png":
            newname[-1] = "jpg"
            newname = str.join(".", newname)
            filename = AugImg_path + "/" + filename
            newname = AugImg_path + "/" + newname
            os.rename(filename, newname)
    print("updated successfully")

    imageList1 = glob.glob(os.path.join(AugImg_path, '*.jpg'))
    # print(imageList1)
    for i in imageList1:
        img = Image.open(i)
        mg = img.convert("RGB")
        mg.save(i)
    print("png->jpg updated")
    copy_allfiles(AugImg_path, path_for_train_img)
    copy_allfiles(AugXml_path, path_for_train_xml)

    f1 = open(TrainInfo_path, 'a')
    for filename in os.listdir(AugImg_path):
        f1.write(filename.rstrip('.jpg'))
        f1.write("\n")
    f1.close()
    print("txt is updated")
    shutil.copyfile(TrainInfo_path, src)
    os.remove(TrainInfo_path)



    print(num_for_all, num_im)
    print(len(list_for_NN), max(list_for_NN))


    # list_for_all_s = np.array(list_for_all_s)
    # np.save('alist_for_all_s_k{}_{}/10.npy'.format(k,i), list_for_all_s)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--InstanceImg_Path", default="information/input_pic", type=str,
                        help="Input instances directory")  ## InstanceImg_path
    parser.add_argument("--InstanceXml_Path", default="information/input_xml", type=str,
                        help="Input instances annotation directory")  ## InstanceXml_path

    parser.add_argument("--OriImg_path", default="data/VOCdevkit/VOC2012", type=str,
                        help="Input instances annotation directory")

    parser.add_argument("--N", default=1, type=int,
                        help="Number of instances to be pasted")

    parser.add_argument("--AugImg_path", default="", type=str,
                        help="Directory for augmented images")  ## AugImg_path
    parser.add_argument("--AugXml_path", default="", type=str,
                        help="Directory for the annotation of augmented images")  ## path_for_instance_xml

    parser.add_argument("--TrainInfo_path", default="information/train.txt", type=str,
                        help="Description of the images to be augmented")

    # parser.add_argument("--lsj", default=True, type=bool, help="if use Large Scale Jittering")
    parser.add_argument("--S_Threshold_1", default=26, type=int,
                        help="Lower size limit of small objects")  ## AugImg_path
    parser.add_argument("--S_Threshold_2", default=64, type=int,
                        help="Upper size limit of small objects")  ## path_for_instance_xml

    parser.add_argument("--w_h_Threshold", default=90, type=int,
                        help="Upper size limit of objects")

    # save_path = './voc_extra_info/voc_data_v2_{}_{}.json'.format(S_Threshold_1, S_Threshold_2)
    # save_path = 'G:/aug/voc_extra_info/voc_data_v2_{}_{}.json'.format(S_Threshold_1, S_Threshold_2)
    # save_path = 'information/voc_data_v2_{}_{}.json'.format(S_Threshold_1, S_Threshold_2)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    list_for_select_pic, list_for_s, list_for_num = generate_instance_info(args)
    ValCopy(args)



