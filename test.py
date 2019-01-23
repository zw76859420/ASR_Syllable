#-*- coding:utf-8 -*-
#author:zhangwei

from speech_model_18 import *
from model_language import ModelLanguage

modelpath_ml = '/home/zhangwei/PycharmProjects/ASR_Thchs30/model_language/'
datapath = '/home/zhangwei/PycharmProjects/ASR_Thchs30/data_list/'
modelpath = '/home/zhangwei/speech_model/'

ms = ModelSpeech(datapath)
ms.load_model(modelpath + 'speech_model_e_0_step_200000.model')
r1 = ms.recognize_speech_fromfile(filename='/home/zhangwei/Desktop/D4_750.wav')
print(r1)
ms.test_model(datapath=datapath , str_dataset='test' , data_count=100)
print('=====================1==================================')

# for i in range(1000 , 20000):
#     if i % 1000 == 0:
#         ms = ModelSpeech(datapath)
#         ms.load_model(modelpath + 'speech_model_e_0_step_' + str(i) + '.model')
#         r1 = ms.recognize_speech_fromfile(filename='/home/zhangwei/Desktop/D4_750.wav')
#         print(r1)
#         ms.test_model(datapath=datapath , str_dataset='test' , data_count=100)
#         print(i)
#         print('=====================1==================================')

# ms = ModelSpeech(datapath)
# ms.load_model(modelpath + 'speech_model_e_0_step_13000.model')
# r1 = ms.recognize_speech_fromfile(filename='/home/zhangwei/Desktop/D4_750.wav')
# print(r1)
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('=====================5==================================')
#
# ms = ModelSpeech(datapath)
# ms.load_model(modelpath + 'speech_model_e_0_step_181000.model')
# r1 = ms.recognize_speech_fromfile(filename='/home/zhangwei/Desktop/D4_750.wav')
# print(r1)
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('=====================5==================================')
#
# ms = ModelSpeech(datapath)
# ms.load_model(modelpath + 'speech_model_e_0_step_196000.model')
# r1 = ms.recognize_speech_fromfile(filename='/home/zhangwei/Desktop/D4_750.wav')
# print(r1)
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('=====================2==================================')
#
# ms = ModelSpeech(datapath)
# ms.load_model(modelpath + 'speech_model_e_0_step_81000.model')
# r1 = ms.recognize_speech_fromfile(filename='/home/zhangwei/Desktop/D4_750.wav')
# print(r1)
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('=====================3==================================')
#
# ms = ModelSpeech(datapath)
# ms.load_model(modelpath + 'speech_model_e_0_step_66000.model')
# r1 = ms.recognize_speech_fromfile(filename='/home/zhangwei/Desktop/D4_750.wav')
# print(r1)
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('=====================4==================================')

# ms = ModelSpeech(datapath)
# ms.load_model(modelpath + 'speech_model_e_0_step_420000.model')
# r1 = ms.recognize_speech_fromfile(filename='/home/zhangwei/Desktop/D4_750.wav')
# print(r1)
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('=====================5==================================')

# ms = ModelSpeech(datapath)
# ms.load_model(modelpath + 'speech_model_e_0_step_196000.model')
# r1 = ms.recognize_speech_fromfile(filename='/home/zhangwei/Desktop/D4_750.wav')
# print(r1)
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('=====================6==================================')
#
# ms = ModelSpeech(datapath)
# ms.load_model(modelpath + 'speech_model_e_0_step_191000.model')
# r1 = ms.recognize_speech_fromfile(filename='/home/zhangwei/Desktop/D4_750.wav')
# print(r1)
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('=====================7==================================')
#
# ms = ModelSpeech(datapath)
# ms.load_model(modelpath + 'speech_model_e_0_step_208000.model')
# r1 = ms.recognize_speech_fromfile(filename='/home/zhangwei/Desktop/D4_750.wav')
# print(r1)
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('=====================8==================================')
#
# ms = ModelSpeech(datapath)
# ms.load_model(modelpath + 'speech_model_e_0_step_118000.model')
# r1 = ms.recognize_speech_fromfile(filename='/home/zhangwei/Desktop/D4_750.wav')
# print(r1)
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('=====================9==================================')

# ms.load_model(modelpath + 'speech_model_e_0_step_126000.model')
# r1 = ms.recognize_speech_fromfile(filename='/home/zhangwei/Desktop/D4_750.wav')
# print(r1)
# print('====================1======================')
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('==========================2========================')
#
# ms.load_model(modelpath + 'speech_model_e_0_step_63000.model')
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('==========================11========================')
#
# ms.load_model(modelpath + 'speech_model_e_0_step_4000.model')
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('==========================3========================')
#
# ms.load_model(modelpath + 'speech_model_e_0_step_53000.model')
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('==========================4========================')


# ms.load_model(modelpath + 'speech_model_e_0_step_26000.model')
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('==========================12========================')
#
# ms.load_model(modelpath + 'speech_model_e_0_step_27000.model')
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('==========================13========================')
#
# ms.load_model(modelpath + 'speech_model_e_0_step_44600.model')
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('==========================5========================')
#
# ms.load_model(modelpath + 'speech_model_e_0_step_44800.model')
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('==========================6========================')
#
# ms.load_model(modelpath + 'speech_model_e_0_step_50000.model')
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('==========================7========================')
#
# ms.load_model(modelpath + 'speech_model_e_0_step_56000.model')
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('==========================8========================')
#
# ms.load_model(modelpath + 'speech_model_e_0_step_31000.model')
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('==========================9========================')
#
# ms.load_model(modelpath + 'speech_model_e_0_step_34000.model')
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('==========================10========================')
#
# ms.load_model(modelpath + 'speech_model_e_0_step_37000.model')
# ms.test_model(datapath=datapath , str_dataset='test' , data_count=2495)
# print('==========================11========================')
