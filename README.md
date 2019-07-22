# ASR_Syllable
=======================基于卷积神经网络的语音识别声学模型的研究========================<br>
## 此项目是对自己研一与研二上之间对于DCNN-CTC学习总结，提出了MCNN-CTC以及Densenet-CTC声学模型，最终实验结果如下所示：<br>
#### 1) Thchs30_TrainingResults<br>
![Thchs30训练以及微调训练曲线](https://github.com/zw76859420/ASR_Syllable/blob/master/training_results/Thchs_Training_Loss.png)
#### 2) Thchs30_Results<br>
![Thchs30实验结果](https://github.com/zw76859420/ASR_Syllable/blob/master/training_results/Thchs_Results.png)
#### 3) Stcmds_Results<br>
![Stcmds实验结果](https://github.com/zw76859420/ASR_Syllable/blob/master/training_results/STCMDS_Results.png)
## 声学模型介绍<br>
#### 1) DCNN-CTC声学模型介绍<br>
  该模型主要是在speech_model-05上进行修改，上述模型主要使用DCNN-CTC构建语音识别声学模型，STcmds 数据集也是仿照该模型进行修改，最后实验结果如上图所示；<br>
#### 2) MCNN-CTC声学模型介绍<br>
  该模型主要是在speech_model_10 脚本上进行实验，最终实验结果可在上图2）所示结果，最终MCNN-CTC总体实验结果相较于DCNN-CTC较好；<br>
#### 3) DenseNet-CTC声学模型介绍<br>
  上述模型主要是在 DenseNet上进行实验，最终实验在Thchs30数据集结果可以达到接近30%左右的CER，具体实验可以自己付尝试一下;<br>
#### 4) Attention-CTC声学模型<br>
  此模型主要在DCNN-CTC基础上，在全连接层进行注意力操作，最终结果相较于其他结果相较于DCNN-CTC可能有提升，具体可以参看speech_model_06脚本；主要算法实验如下所示：<br>
  NN(Attention)-CTC:<br>
        # dense1 = Dense(units=512, activation='relu', use_bias=True, kernel_initializer='he_normal')(reshape)<br>
        # attention_prob = Dense(units=512, activation='softmax', name='attention_vec')(dense1)<br>
        # attention_mul = multiply([dense1, attention_prob])<br>
        #<br>
        # dense1 = BatchNormalization(epsilon=0.0002)(attention_mul)<br>
        # dense1 = Dropout(0.3)(dense1)<br>
## 迁移学习<br>
  Retraining(重新训练)主要对初始模型进行进一步微调，可进一步提升初始模型的准确率，具体训练脚本可参看 train_modelSpeech 脚本，本文主要针对全部网路层进行微调，实验结果相较于初始模型可进一步提升，具体实验结果可参看图1)<br>
## 论文引用<br>
  W Zhang, M H Zhai, Z L Huang, et al. Towards End-to-End Speech Recognition with Deep Multipath Convolutional Neural Networks[C]. https://doi.org/10.1007/978-3-030-27529-7_29<br>
## 参考项目连接<br>
[个人博客](https://blog.csdn.net/Xwei1226 "悬停显示") 包含自己近期的学习总结<br>
[参考链接](https://github.com/nl8590687/ASRT_SpeechRecognition "悬停显示")<br>
[ASR_WORD](https://github.com/zw76859420/ASR_WORD "悬停显示")以字为建模单元构建语音识别声学模型<br>

