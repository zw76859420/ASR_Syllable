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
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image , IMAGE_SIZE , IMAGE_SIZE)
                # cv2.imwrite('/home/zhangwei/1.jpg' , image)
                # cv2.imshow("image" ,image)
                # cv2.waitKey(10)
                images.append(image)
                labels.append(pathname)
    return images , labels

def load_dataset(pathname):
    images , labels = read_path(pathname)
    images = np.array(images)
    # print(labels)
    # print(images.shape)

    labels = np.array([0 if label.endswith('ywl') else 1 for label in labels])
    # print(labels)
    # print(labels.shape , type(labels))
    # for label in labels:
    #     if label.endswith('ywl'):
    #         label = 0
    #     else:
    #         label = 1
    #     print(label)

    return images , labels

if __name__ == "__main__":
    pathname = '/home/zhangwei/data/'
    a , b = load_dataset(pathname)
    print(a)




