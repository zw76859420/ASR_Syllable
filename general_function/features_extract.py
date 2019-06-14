#-*- coding:utf-8 -*-
#author:zhangwei

"""
   该脚本用于提取语音特征，包括MFCC、FBANK以及语谱图特征；
   该脚本是对标签数据进行处理；
"""

from python_speech_features import mfcc, delta, logfbank
import wave
import numpy as np
from scipy.fftpack import fft

def read_wav_data(filename):
    '''
    获取文件数据以及采样频率；
    输入为文件位置，输出为wav文件数学表示和采样频率；
    '''
    wav = wave.open(filename, 'rb')
    num_frames = wav.getnframes()
    num_channels = wav.getnchannels()
    framerate = wav.getframerate()
    str_data = wav.readframes(num_frames)
    wav.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, num_channels
    wave_data = wave_data.T
    return wave_data, framerate

def get_mfcc_feature(wavsignal, fs):
    '''
    输入为wav文件数学表示和采样频率，输出为语音的MFCC特征+一阶差分+二阶差分；
    '''
    feat_mfcc = mfcc(wavsignal, fs)
    print(feat_mfcc)
    feat_mfcc_d = delta(feat_mfcc, 2)
    feat_mfcc_dd = delta(feat_mfcc_d, 2)
    wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
    return wav_feature

def get_fbank_feature(wavsignal, fs):
    '''
    输入为wav文件数学表示和采样频率，输出为语音的FBANK特征+一阶差分+二阶差分；
    '''
    feat_fbank = logfbank(wavsignal, fs, nfilt=40)
    feat_fbank_d = delta(feat_fbank, 2)
    feat_fbank_dd = delta(feat_fbank_d, 2)
    wav_feature = np.column_stack((feat_fbank, feat_fbank_d, feat_fbank_dd))
    return wav_feature

# def get_frequency_feature(wavsignal , fs):
#     time_window = 25
#     window_length = fs / 1000 * time_window
#     wav_arr = np.array(wavsignal)
#     wav_length = wav_arr.shape[1]
#     range0_end = (int(len(wavsignal[0]) / fs * 1000 - time_window) // 10) // 2
#     data_input = np.zeros((range0_end , 200) , dtype=np.float)
#     data_line = np.zeros((1,400), dtype=np.float)
#     # print(range0_end)
#     for i in range(0 , range0_end):
#         p_start = i * 160
#         p_end = p_start + 400
#         data_line = wav_arr[0 , p_start : p_end]
#         # print(data_line.shape , i)
#         data_line = data_line * w
#         data_line = np.abs(fft(data_line)) / wav_length
#         data_input[i] = data_line[0 : 200]
#     data_input = np.log(data_input + 1)
#     return data_input

def get_frequency_feature(wavsignal, fs):
    '''
    输入为wav文件数学表示和采样频率,输出为语谱图特征，特征维度是200；
    '''
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))
    time_window = 25
    wav_array = np.array(wavsignal)
    wav_length = wav_array.shape[1]
    first2end = (int(len(wavsignal[0]) / fs * 1000 - time_window) // 10) // 2
    data_input = np.zeros(shape=[first2end, 200], dtype=np.float)
    for i in range(0, first2end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_array[0, p_start:p_end]
        data_line = data_line * w
        data_line = np.abs(fft(data_line)) / wav_length
        data_input[i] = data_line[0: 200]
    data_input = np.log(data_input + 1)
    return data_input

def main(file_path):
    wav , fs = read_wav_data(file_path)
    features = get_frequency_feature(wav , fs)
    # features = get_fbank_feature(wav , fs)
    return features


if __name__ == '__main__':
    file_path = '/home/zhangwei/Desktop/D4_751.wav'
    # a = main(file_path)
    # print(a.shape)
    a, b = read_wav_data(file_path)
    print(a.shape, b)