#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: nl8590687
asrserver测试专用客户端

'''

import requests
from general_function.features_extract import *

# url = 'http://10.42.0.106/'

url = 'http://127.0.0.1:20000/'

token = 'qwertasd'

wavsignal, fs=read_wav_data('/home/zhangwei/Desktop/D4_751.wav')

print(wavsignal, fs)

datas={'token': token, 'fs': fs, 'wavs': wavsignal}

r = requests.post(url, datas)

r.encoding ='utf-8'

print(r.text)