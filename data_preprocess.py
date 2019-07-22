#-*- coding:utf-8 -*-
#author:zhangwei

import os
import sys
import numpy as np
import cv2

IMAGE_SIZE = 128

def resize_image(image , height=IMAGE_SIZE , width=IMAGE_SIZE):
    top , bottom , left , right = (0 , 0 , 0 , 0)
    h , w , channels = image.shape
    # print(h , w)
    longest_edge = max(h , w)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
        # print(dw , left , right)
    else:
        pass

    #print(top , bottom , left , right)

    BLACK = [0 , 0 , 0]
    constant = cv2.copyMakeBorder(image , top , bottom , left , right , cv2.BORDER_CONSTANT , value=BLACK)

    return cv2.resize(constant , (height , width))

images = []
labels = []

def read_path(pathname):
    for dir_item in os.listdir(pathname):
        full_path = os.path.abspath(os.path.join(pathname , dir_item))
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            if dir_item.endswith('.bmp'):
                image = cv2.imread(full_path)
                image = resize_image(image , IMAGE_SIZE , IMAGE_SIZE)
                # cv2.imwrite('/home/zhangwei/1.jpg' , image)
                images.append(image)
                # print(pathname)
                labels.append(pathname)
    return images , labels

def load_dataset(pathname):
    images , labels = read_path(pathname)
    images = np.array(images)
    labels = np.array([0 if label.endswith('0') else 1 for label in labels])
    return images , labels

if __name__ == '__main__':
    pathname = '/home/zhangwei/data/ScanKnife/'
    load_dataset(pathname)


'''
def resize_image(imagepath , height=64 , width=64):
    # image = cv2.imread(imagepath)
    top , bottom , left , right = (0 , 0 , 0 , 0)
    h , w , channels = imagepath.shape
    # print(image.shape)
    # print(h)
    longest_edge = max(h , w)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
        # print(top , bottom)
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
        # print(right)
    else:
        pass

    BLACK = [0 , 0 , 0]
    constant = cv2.copyMakeBorder(imagepath , top , bottom , left , right , cv2.BORDER_CONSTANT , value=BLACK)
    return cv2.resize(constant , (height , width))

images = []
labels = []

def read_path(pathname):
    for i in os.listdir(pathname):
        j = os.path.abspath(os.path.join(pathname , i))
        # print(pathname)
        if os.path.isdir(j):
            read_path(j)
        else:
            if i.endswith('.bmp'):
                image = cv2.imread(j)
                image = resize_image(image)
                # cv2.imwrite('/home/zhangwei/1.bmp' , image)
                images.append(image)
                # print(pathname)
                labels.append(pathname)
    return images , labels

def load_data(pathname):
    images , labels = read_path(pathname)
    images = np.array(images , dtype='float32')
    labels_re = []
    for label in labels:
        if label.endswith('1'):
            label = '1'
            labels_re.append(label)
        elif label.endswith('0'):
            label = '0'
            labels_re.append(label)
        else:
            pass

    labels_re = np.array(labels_re , dtype='int32')
    return images , labels_re

if __name__ == '__main__':
    pathname = '/home/zhangwei/data/ScanKnife/'
    a , b = load_data(pathname)
    # print(a[0] , b[0])
    # imagepath = '/home/zhangwei/T005H1D1S002.bmp'
    # image = resize_image(imagepath)
    # cv2.imwite
'''



