#-*- coding:utf-8 -*-
#author:zahngwei
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
from keras.layers import Dense , Dropout , Input , Reshape , multiply
from keras.layers import Conv2D , MaxPooling2D , Lambda , Activation
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import SGD , Adadelta , Adam
from readdata_07 import DataSpeech


class ModelSpeech():  # 语音模型类
    def __init__(self, datapath):
        '''
        初始化
        默认输出的拼音的表示大小是1422，即1421个拼音+1个空白块
        '''
        MS_OUTPUT_SIZE = 1422
        self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE  # 神经网络最终输出的每一个字符向量维度的大小
        # self.BATCH_SIZE = BATCH_SIZE # 一次训练的batch
        self.label_max_string_length = 64
        self.AUDIO_LENGTH = 1600
        self.AUDIO_FEATURE_LENGTH = 200
        self._model, self.base_model = self.CreateModel()

        self.datapath = datapath
        self.slash = '/'

        if (self.slash != self.datapath[-1]):  # 在目录路径末尾增加斜杠
            self.datapath = self.datapath + self.slash

    def CreateModel(self):
        '''
        定义CNN/LSTM/CTC模型，使用函数式模型
        输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
        隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2
        隐藏层：全连接层
        输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数，
        CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出

        '''
        # 每一帧使用13维mfcc特征及其13维一阶差分和13维二阶差分表示，最大信号序列长度为1500
        input_data = Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))

        layer_h1 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(input_data)  # 卷积层
        layer_h1 = BatchNormalization(epsilon=0.0002)(layer_h1)
        # layer_h1 = Dropout(0.1)(layer_h1)
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层
        # layer_h3 = Dropout(0.2)(layer_h2) # 随机中断部分神经网络连接，防止过拟合
        # layer_h3 = Dropout(0.1)(layer_h3)
        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3)  # 卷积层
        # layer_h4 = Dropout(0.2)(layer_h4)
        layer_h5 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)  # 池化层

        # layer_h6 = Dropout(0.2)(layer_h6)
        layer_h7 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal')(layer_h6)  # 卷积层
        # layer_h7 = Dropout(0.3)(layer_h7)
        layer_h8 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal')(layer_h7)  # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)  # 池化层

        # layer_h9 = Dropout(0.3)(layer_h9)
        layer_h10 = Conv2D(256, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal')(layer_h9)  # 卷积层
        # layer_h10 = Dropout(0.4)(layer_h10)
        layer_h11 = Conv2D(256, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal')(layer_h10)  # 卷积层
        layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11)  # 池化层

        # layer_h12 = Dropout(0.4)(layer_h12)
        layer_h13 = Conv2D(256, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal')(layer_h12)  # 卷积层
        # layer_h13 = Dropout(0.5)(layer_h13)
        layer_h14 = Conv2D(256, (3, 3), use_bias=True, activation='relu', padding='same',kernel_initializer='he_normal')(layer_h13)  # 卷积层
        layer_h15 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14)  # 池化层

        # test=Model(inputs = input_data, outputs = layer_h12)
        # test.summary()

        layer_h16 = Reshape((200, 6400))(layer_h15)  # Reshape层
        # layer_h5 = LSTM(256, activation='relu', use_bias=True, return_sequences=True)(layer_h4) # LSTM层
        # layer_h6 = Dropout(0.2)(layer_h5) # 随机中断部分神经网络连接，防止过拟合
        # layer_h16 = Dropout(0.5)(layer_h16)
        layer_h17 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16)  # 全连接层
        # layer_h17 = Dropout(0.5)(layer_h17)
        layer_h18 = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(layer_h17)  # 全连接层

        y_pred = Activation('softmax', name='Activation0')(layer_h18)
        model_data = Model(inputs=input_data, outputs=y_pred)
        # model_data.summary()

        labels = Input(name='the_labels', shape=[self.label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer

        # layer_out = Lambda(ctc_lambda_func,output_shape=(self.MS_OUTPUT_SIZE, ), name='ctc')([y_pred, labels, input_length, label_length])#(layer_h6) # CTC
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length])

        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        # model.summary()

        # clipnorm seems to speeds up convergence
        # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        # opt = Adadelta(lr = 0.01, rho = 0.95, epsilon = 1e-06)
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=10e-8)
        # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)

        # captures output of softmax so we can decode the output during visualization
        test_func = K.function([input_data], [y_pred])

        # print('[*提示] 创建模型成功，模型编译成功')
        print('[*Info] Create Model Successful, Compiles Model Successful. ')
        return model, model_data

    def ctc_lambda_func(self , args):
        y_pred , labels , input_length , label_length = args
        y_pred = y_pred[: , : , :]
        return K.ctc_batch_cost(y_true=labels , y_pred=y_pred , input_length=input_length , label_length=label_length)

    def train_model(self , datapath , epoch=4 , save_step=2000 , batch_size=8):
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
                self.test_model(datapath=self.datapath , str_dataset='train' , data_count=4)
                self.test_model(datapath=self.datapath , str_dataset='dev' , data_count=16)

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

                pre = self.predict(data_input=data_input , input_len=data_input.shape[0] // 32)
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

    def save_model(self , filename='/home/zhangwei/speech_model/speech_model' , comments=''):
        self._model.save_weights(filename + comments + '.model')
        self.base_model.save_weights(filename + comments + '.model.base')
        f = open('steps24.txt' , 'w')
        f.write(filename + comments)
        f.close()

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
        input_length = input_length // 32
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
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    set_session(tf.Session(config=config))

    datapath = '/home/zhangwei/PycharmProjects/ASR_Thchs30/data_list/'
    speech = ModelSpeech(datapath=datapath)
    speech.creat_model()
    # speech.train_model(datapath=datapath)