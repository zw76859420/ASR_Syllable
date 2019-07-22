#-*- coding:utf-8 -*-
#author:zhangwei

import numpy as np
from general_function.file_wav import *
from general_function.file_dict import *
from general_function.feature_extract import *

import random

class DataSpeech():

    def __init__(self , path , type):
        self.datapath = path
        self.type = type

        self.slash = '/'
        if self.slash != self.datapath[-1]:
            self.datapath = self.datapath + self.slash

        self.dic_wavlist_thchs30 = {}
        self.dic_symbollist_thchs30 = {}
        self.symbolnum = 0
        self.datanum = 0
        self.wavs_data = []
        self.list_wavnum_thchs30 = []
        self.list_symbolnum_thchs30 = []
        self.load_datalist()
        self.list_symbol = self.get_symbollist()

        pass

    def load_datalist(self):
        if self.type == 'train':
            filename_wavlist_thchs30 = 'thchs30' + self.slash + 'train.wav.lst'
            filename_symbollist_thchs30 = 'thchs30' + self.slash + 'train.syllable.txt'
        elif self.type == 'dev':
            filename_wavlist_thchs30 = 'thchs30' + self.slash + 'cv.wav.lst'
            filename_symbollist_thchs30 = 'thchs30' + self.slash + 'cv.syllable.txt'
        elif self.type == 'test':
            filename_wavlist_thchs30 = 'thchs30' + self.slash + 'test.wav.lst'
            filename_symbollist_thchs30 = 'thchs30' + self.slash + 'test.syllable.txt'
        else:
            pass
        self.dic_wavlist_thchs30 , self.list_wavnum_thchs30 = get_wav_list(self.datapath + filename_wavlist_thchs30)
        self.dic_symbollist_thchs30 , self.list_symbolnum_thchs30 = get_wav_symbol(self.datapath + filename_symbollist_thchs30)
        self.datanum = self.get_datanum()

    def get_datanum(self):
        num_wavlist_thchs30 = len(self.dic_wavlist_thchs30)
        num_symbollist_thchs30 = len(self.dic_symbollist_thchs30)
        if num_wavlist_thchs30 == num_symbollist_thchs30:
            datanum = num_wavlist_thchs30
        else:
            datanum = -1
        return datanum

    def get_data(self , n_start):
        filename = self.dic_wavlist_thchs30[self.list_wavnum_thchs30[n_start]]
        list_symbol = self.dic_symbollist_thchs30[self.list_symbolnum_thchs30[n_start]]
        wavsignal , fs = read_wav_data(self.datapath + filename)
        feat_out = []
        for i in list_symbol:
            n = self.symbol_to_num(i)
            feat_out.append(n)
        data_input = get_frequency_feature(wavsignal , fs)
        data_input = data_input.reshape(data_input.shape[0] , data_input.shape[1] , 1)
        data_label = np.array(feat_out)
        return data_input , data_label

    def data_generator(self , batch_size=8 , audio_length=1600):
        labels = []
        for i in range(0 , batch_size):
            labels.append([0.0])

        labels = np.array(labels , dtype=np.float)

        while True:
            X = np.zeros([batch_size , audio_length , 200 , 1] , dtype=np.float)
            y = np.zeros([batch_size , 64] , dtype=np.int16)

            input_length = []
            label_length = []

            for i in range(batch_size):
                ran_num = random.randint(0 , self.datanum - 1)
                data_input , data_labels = self.get_data(ran_num)
                input_length.append([data_input.shape[0] // 16 + data_input.shape[0] % 8])
                X[i , 0 : len(data_input)] = data_input
                y[i , 0 : len(data_labels)] = data_labels
                label_length.append([len(data_labels)])

            label_length = np.array(label_length)
            input_length = np.array(input_length)

            yield [X , y , input_length , label_length] , labels
        pass

    def get_symbollist(self):
        list_symbol = []
        with open('dict.txt' , 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                res = line.split()
                list_symbol.append(res[0])
        list_symbol.append('_')
        self.symbolnum = len(list_symbol)
        return list_symbol


    def symbol_to_num(self , symbol):
        if symbol != '':
            return self.list_symbol.index(symbol)
        else:
            return self.symbolnum

    def get_symbol_num(self):
        return len(self.list_symbol)



if __name__ == '__main__':
    datapath = '/home/zhangwei/PycharmProjects/ASR_Thchs30/data_list/'
    Data = DataSpeech(path=datapath , type='train')
    data_input , data_labels = Data.get_data(0)
    print(data_labels.shape)

    # aa = Data.data_generator()
    # for i in aa:
    #     print i[0][2]