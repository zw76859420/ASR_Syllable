# -*- coding:utf-8 -*-
#author:zhangwei

from sk_mcnn_01 import ModelSpeech
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.90
set_session(tf.Session(config=config))

modelpath = '/home/zhangwei/speech_model_01/'
datapath = '/home/zhangwei/PycharmProjects/ASR_Thchs30/data_list/'

speech = ModelSpeech(datapath=datapath)
speech.load_model(filename=modelpath + 'speech_model_e_0_step_206000.model')
speech.train_model(datapath=datapath , save_step=500)