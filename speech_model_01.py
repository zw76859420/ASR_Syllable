#-*- coding:utf-8 -*-
#author:zhangwei

from general_function.file_wav import *
from general_function.file_wav import *
from general_function.file_dict import *
from general_function.feature_extract import *
from general_function.edit_distance import *

import keras as kr
import numpy as np
import random

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense , Dropout , Input , Reshape
from keras.layers import Conv2D , MaxPooling2D , Lambda , Activation
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import SGD , Adadelta

from readdata_01 import DataSpeech

class ModelSpeech():
    def __init__(self , datapath):
        MS_OUTPUT_SIZE = 1422
        self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE
        self.label_max_string_length = 64
        self.AUDIO_LENGTH = 1600
        self.AUDIO_FEATURE_LENGTH = 200
        self.datapath = datapath
        self._model , self.base_model = self.creat_model()

        self.slash = '/'
        if self.datapath[-1] != self.slash:
            self.datapath = self.datapath + self.slash
        pass

    def creat_model(self):

        input_data = Input(shape=[self.AUDIO_LENGTH , self.AUDIO_FEATURE_LENGTH , 1] , name='Input')
        conv1 = Conv2D(filters=32 , kernel_size=[3,3] , padding='same' , activation='relu' , use_bias=True , kernel_initializer='he_normal')(input_data)
        conv1 = BatchNormalization(epsilon=0.0002)(conv1)
        conv2 = Conv2D(filters=32 , kernel_size=[3,3] , padding='same' , activation='relu' , use_bias=True , kernel_initializer='he_normal')(conv1)
        conv2 = BatchNormalization(epsilon=0.0002)(conv2)
        maxpool1 = MaxPooling2D(pool_size=[2,2] , strides=None , padding='valid')(conv2)
        maxpool1 = Dropout(0.2)(maxpool1)

        conv3 = Conv2D(filters=64 , kernel_size=[3,3] , padding='same' , activation='relu' , use_bias=True , kernel_initializer='he_normal')(maxpool1)
        conv3 = BatchNormalization(epsilon=0.0002)(conv3)
        conv4 = Conv2D(filters=64 , kernel_size=[3,3] , padding='same' , activation='relu' , use_bias=True , kernel_initializer='he_normal')(conv3)
        conv4 = BatchNormalization(epsilon=0.0002)(conv4)
        maxpool2 = MaxPooling2D(pool_size=[2,2] , strides=None , padding='valid')(conv4)
        maxpool2 = Dropout(0.2)(maxpool2)

        layer_reshape = Reshape([400,3200])(maxpool2)
        dense1 = Dense(units=1024 , activation='relu' , use_bias=True , kernel_initializer='he_normal')(layer_reshape)
        dense1 = BatchNormalization(epsilon=0.0002)(dense1)
        dense1 = Dropout(0.3)(dense1)
        dense2 = Dense(units=self.MS_OUTPUT_SIZE , use_bias=True , kernel_initializer='he_normal')(dense1)
        y_pred = Activation(activation='softmax' , name='activation')(dense2)
        model_data = Model(inputs=input_data , outputs=y_pred)

        # model_data.summary()
        # plot_model(model_data , '/home/zhangwei/model.png')

        labels = Input(shape=[self.label_max_string_length] , name='labels' , dtype='float32')
        input_length = Input(shape=[1] , name='input_length' , dtype='int64')
        label_length = Input(shape=[1] , name='label_length' , dtype='int64')

        loss_out = Lambda(self.ctc_lambda_func , output_shape=[1,] , name='ctc')([y_pred , labels , input_length , label_length])
        model = Model(inputs=[input_data , labels , input_length , label_length] , outputs=loss_out)

        ada_d = Adadelta(lr=0.01 , rho=0.95 , epsilon=1e-6)
        model.compile(optimizer=ada_d , loss={'ctc' : lambda y_true , y_pred : y_pred})

        print('==========================模型创建成功=================================')
        return model , model_data


    def ctc_lambda_func(self , args):
        y_pred , labels , input_length , label_length = args
        y_pred = y_pred[: , : , :]
        return K.ctc_batch_cost(y_true=labels , y_pred=y_pred , input_length=input_length , label_length=label_length)


    def train_model(self , datapath , epoch=4 , save_step=1000 , batch_size=8):
        data = DataSpeech(datapath , 'train')
        num_data = data.get_datanum()
        yielddatas = data.data_generator(batch_size , self.AUDIO_LENGTH)
        for epoch in range(epoch):
            print('[*running] train epoch %d .' % epoch)
            n_step = 0
            while True:
                try:
                    print('[*message] epoch %d , Having training data %d+' % (epoch , n_step * save_step))
                    self._model.fit_generator(yielddatas , save_step)
                    n_step += 1
                except StopIteration:
                    print('======================Error StopIteration==============================')
                    break
                self.save_model(comments='_e_' + str(epoch) + '_step_' + str(n_step * save_step))
                self.test_model(datapath=self.datapath , str_dataset='train' , data_count=8)
                self.test_model(datapath=self.datapath , str_dataset='dev' , data_count=8)

    def save_model(self , filename='/home/zhangwei/speech_model/speech_model' , comments=''):
        self._model.save_weights(filename + comments + '.model')
        self.base_model.save_weights(filename + comments + '.model.base')
        f = open('steps24.txt' , 'w')
        f.write(filename + comments)
        f.close()

    def load_model(self , filename='model_speech_e_0_step_16000.model'):
        self._model.load_weights(filename)
        self.base_model.load_weights(filename + '.base')


    def test_model(self , datapath='' , str_dataset='dev' , data_count=1):
        data = DataSpeech(self.datapath , str_dataset)
        num_data = data.get_datanum()
        # print num_data
        if data_count <=0 and data_count > num_data:
            data_count = num_data
        try:
            ran_num = random.randint(0 , num_data - 1)
            words_num = 0.
            word_error_num = 0.
            for i in range(data_count):
                data_input , data_labels = data.get_data((ran_num + i) % num_data)
                # print data_input
                num_bias = 0
                while data_input.shape[0] > self.AUDIO_LENGTH:
                    print('[*Error] data input is too long %d' % ((ran_num + i) % num_data))
                    num_bias += 1
                    data_input , data_labels = data.get_data((ran_num + i + num_bias) % num_data)

                pre = self.predict(data_input=data_input , input_len=data_input.shape[0] //4)
                words_n = data_labels.shape[0]
                words_num += words_n
                edit_distance = get_edit_distance(data_labels , pre)
                if edit_distance <= words_n:
                    word_error_num += edit_distance
                else:
                    word_error_num += words_n
            # print type(words_num)
            print('[*Test Result] Speech Recognition ' + str_dataset + ' set word error ratio : ' + str(word_error_num / words_num * 100) , '%')


        except StopIteration:
            print('=======================Error StopIteration 01======================')

    def predict(self , data_input , input_len):
        batch_size = 1
        in_len = np.zeros((batch_size) , dtype=np.int32)
        in_len[0] = input_len
        x_in = np.zeros(shape=[batch_size , 1600 , self.AUDIO_FEATURE_LENGTH , 1] , dtype=np.float)
        for i in range(batch_size):
            x_in[i , 0 : len(data_input)] = data_input
        base_pred = self.base_model.predict(x=x_in)
        base_pred = base_pred[: , : , :]
        r = K.ctc_decode(base_pred , in_len , greedy=True , beam_width=100 , top_paths=1)
        r1 = K.get_value(r[0][0])
        r1 = r1[0]
        return r1

    def redognize_speech(self , wavsignal , fs):
        data_input = get_frequency_feature(wavsignal , fs)
        input_length = len(data_input)
        input_length = input_length //4
        data_input = np.array(data_input , dtype=np.float)
        data_input = data_input.reshape(data_input.shape[0] , data_input.shape[1] , 1)
        r1 = self.predict(data_input , input_length)
        # print r1
        list_symbol_dic = get_list_symbol(self.datapath)
        r_str = []
        for i in r1:
            r_str.append(list_symbol_dic[i])
        return r_str

    def recognize_speech_fromfile(self , filename):
        wavsignal , fs = read_wav_data(filename)
        r = self.redognize_speech(wavsignal , fs)
        return r


if __name__ == '__main__':
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))

    filename = '/home/zhangwei/Desktop/A11_101.wav'
    wavsignal, fs = read_wav_data(filename)
    datapath = '/home/zhangwei/PycharmProjects/ASR_Thchs30/data_list/'
    speech = ModelSpeech(datapath=datapath)
    speech.train_model(datapath=datapath)
    # speech.creat_model()