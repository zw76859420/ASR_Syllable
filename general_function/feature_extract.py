#-*- coding:utf-8 -*-
#author:zhangwei

import wave
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from scipy.fftpack import fft

def read_wav_data(filename):
    wav = wave.open(filename, 'rb')
    num_frames = wav.getnframes()
    num_channel = wav.getnchannels()
    framerate = wav.getframerate()
    str_data = wav.readframes(num_frames)
    wav.close()
    wav_data = np.fromstring(str_data , dtype=np.short)
    wav_data.shape = -1 , num_channel
    wav_data = wav_data.T
    return wav_data , framerate

'''
定义汉明窗
'''
x = np.linspace(0 , 400 - 1 , 400 , dtype=np.int64)
w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))

def get_frequency_feature(wavsignal , fs):
    time_window = 25
    window_length = fs / 1000 * time_window
    wav_arr = np.array(wavsignal)
    wav_length = wav_arr.shape[1]
    range0_end = int(len(wavsignal[0]) / fs * 1000 - time_window) // 10
    data_input = np.zeros((range0_end , 200) , dtype=np.float)
    data_line = np.zeros((1,400), dtype=np.float)
    # print(range0_end)
    for i in range(0 , range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[0 , p_start : p_end]
        # print(data_line.shape , i)
        data_line = data_line * w
        data_line = np.abs(fft(data_line)) / wav_length
        data_input[i] = data_line[0 : 200]
    data_input = np.log(data_input + 1)
    return data_input

if __name__ == '__main__':
    filename = '/home/zhangwei/Desktop/D4_750.wav'
    filename_01 = '/home/zhangwei/01.wav'
    wavsignal, fs = read_wav_data(filename)
    # print(wavsignal[0].shape)
    wavsignal = get_frequency_feature(wavsignal, fs)
    print(wavsignal.shape)

    # freimg = get_frequency_feature(wavsignal, fs)
    # freimg = freimg.T
    #
    # plt.subplot(111)
    # plt.imshow(freimg)
    # plt.colorbar(cax=None , ax=None , shrink=0.5)
    # plt.show()
    # print(len(wavsignal[0]))
    # plt.specgram(wavsignal , NFFT=1024 , Fs=16000)
    # plt.colorbar(cax=None , ax=None , shrink=0.5)
    # plt.show()