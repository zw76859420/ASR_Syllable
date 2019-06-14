#-*- coding:utf-8 -*-
#author:zhangwei

import numpy as np
import wave as wav
from scipy.io import wavfile as wav
from python_speech_features import mfcc
import matplotlib.pyplot as plt
from general_function.features_extract import read_wav_data , main , get_fbank_feature


def pinzhen(file_path , n_input , n_context):
    wav_data , fs = read_wav_data(file_path)
    origin_inputs = mfcc(wav_data , samplerate=fs , numcep=n_input)
    '''
       跳帧选择需要的特征;
    '''
    # print(origin_inputs)
    '''
       这里涉及跳帧处理，隔一帧，取一列特征;
    '''
    # origin_inputs = origin_inputs[::2]
    # print(origin_inputs)
    '''
       初始化最后提取的总特征维度;
    '''
    train_inputs = np.zeros(shape=(origin_inputs.shape[0] , n_input + 2 * n_input * n_context))
    '''
       初始化需要填补的MFCC特征;
    '''
    empty_mfcc = np.zeros((n_input))

    time_slices = range(train_inputs.shape[0])
    '''
       设置初始需要拼帧与未来需要拼帧的初始位置;
    '''
    context_past_min = time_slices[0] + n_context
    context_future_max = time_slices[-1] - n_context


    for time_slice in time_slices:
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = origin_inputs[max(0 , time_slice - n_context) : time_slice]
        need_empty_future = max(0 , (time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = origin_inputs[time_slice + 1:time_slice + n_context + 1]

        if need_empty_past:
            past = np.concatenate((empty_source_past , data_source_past))
        else:
            past = data_source_past

        if need_empty_future:
            future = np.concatenate((data_source_future , empty_source_future))
        else:
            future = data_source_future

        past = np.reshape(past , n_context * n_input)
        now = origin_inputs[time_slice]
        future = np.reshape(future , n_context * n_input)

        train_inputs[time_slice] = np.concatenate((past , now , future))
    '''
       可以做一下均值归一化，将数据服从正太分布标准，减去均值再除以方差
    '''
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    return train_inputs

def pinzhen_spectrogram(file_path , n_context=1):
    origin_inputs = main(file_path)
    # print(origin_inputs.shape)
    '''
       跳帧选择需要的特征;
    '''
    n_input = origin_inputs.shape[1]
    '''
       这里涉及跳帧处理，隔一帧，取一列特征;
    '''
    # origin_inputs = origin_inputs[::2]
    # print(origin_inputs)
    '''
       初始化最后提取的总特征维度;
    '''
    train_inputs = np.zeros(shape=(origin_inputs.shape[0] , n_input + 2 * n_input * n_context))
    '''
       初始化需要填补的MFCC特征;
    '''
    empty_mfcc = np.zeros((n_input))

    time_slices = range(train_inputs.shape[0])
    '''
       设置初始需要拼帧与未来需要拼帧的初始位置;
    '''
    context_past_min = time_slices[0] + n_context
    context_future_max = time_slices[-1] - n_context


    for time_slice in time_slices:
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = origin_inputs[max(0 , time_slice - n_context) : time_slice]
        need_empty_future = max(0 , (time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = origin_inputs[time_slice + 1:time_slice + n_context + 1]

        if need_empty_past:
            past = np.concatenate((empty_source_past , data_source_past))
        else:
            past = data_source_past

        if need_empty_future:
            future = np.concatenate((data_source_future , empty_source_future))
        else:
            future = data_source_future

        past = np.reshape(past , n_context * n_input)
        now = origin_inputs[time_slice]
        future = np.reshape(future , n_context * n_input)

        train_inputs[time_slice] = np.concatenate((past , now , future))
    '''
       可以做一下均值归一化，将数据服从正太分布标准，减去均值再除以方差;
    '''
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    return train_inputs

def pinzhen_fbank(file_path , n_context=1):
    wav_data , fs = read_wav_data(file_path)
    origin_inputs = get_fbank_feature(wav_data , fs)
    # print(origin_inputs.shape)
    '''
       跳帧选择需要的特征;
    '''
    n_input = origin_inputs.shape[1]
    '''
       这里涉及跳帧处理，隔一帧，取一列特征;
    '''
    origin_inputs = origin_inputs[::2]
    # print(origin_inputs)
    '''
       初始化最后提取的总特征维度;
    '''
    train_inputs = np.zeros(shape=(origin_inputs.shape[0] , n_input + 2 * n_input * n_context))
    '''
       初始化需要填补的MFCC特征;
    '''
    empty_mfcc = np.zeros((n_input))

    time_slices = range(train_inputs.shape[0])
    '''
       设置初始需要拼帧与未来需要拼帧的初始位置;
    '''
    context_past_min = time_slices[0] + n_context
    context_future_max = time_slices[-1] - n_context


    for time_slice in time_slices:
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = origin_inputs[max(0 , time_slice - n_context) : time_slice]
        need_empty_future = max(0 , (time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = origin_inputs[time_slice + 1:time_slice + n_context + 1]

        if need_empty_past:
            past = np.concatenate((empty_source_past , data_source_past))
        else:
            past = data_source_past

        if need_empty_future:
            future = np.concatenate((data_source_future , empty_source_future))
        else:
            future = data_source_future

        past = np.reshape(past , n_context * n_input)
        now = origin_inputs[time_slice]
        future = np.reshape(future , n_context * n_input)

        train_inputs[time_slice] = np.concatenate((past , now , future))
    '''
       可以做一下均值归一化，将数据服从正太分布标准，减去均值再除以方差;
    '''
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

    return train_inputs


if __name__ == '__main__':
    file_path = '/home/zhangwei/Desktop/D4_751.wav'
    # a = pinzhen_spectrogram(file_path , n_context=1)
    freimg = pinzhen_fbank(file_path ,n_context=1)
    freimg = freimg

    plt.subplot(111)
    plt.imshow(freimg)
    plt.colorbar(cax=None, ax=None, shrink=0.5)
    plt.show()
    # a = pinzhen_fbank(file_path)
    # print(a.shape)
    print(freimg.shape)