# ASR_Syllable
  采用端到端方法构建声学模型，以音节为建模单元，采用DCNN-CTC网络结构，希望集众人力量改进以音节为建模单元的声学模型。
## 配置环境要求
python 3.x <br>
keras 2.0 <br>
Tensorflow 1.8.0 <br>
numpy <br>
wave <br>
python_speech_features <br>
random <br>
os <br>
scipy <br>
difflib<br>
## THCHS-30数据
  程序的实验数据可以从[Thchs30](http://www.openslr.org/18/ "悬停显示")下载，在此感谢清华大学对语音识别领域的贡献； 下载完成解压后放在程序的同级目录； 程序不仅是对中文语音识别声学模型以字为建模单元的的构建，而且为英文语音识别以字为建模单元提供一定的指导意义。
## 算法框架
### CTC损失函数：
  采用DCNN-CTC构建语音识别声学模型；<br>
  CTC损失函数介绍可以观看本人的博客；<br>
  [(1)CTC原理介绍](https://blog.csdn.net/Xwei1226/article/details/80969250 "悬停显示");<br>
  [(2)CTC损失函数推导](https://blog.csdn.net/Xwei1226/article/details/80889818 "悬停显示");<br>
### 网络构建；
  (1)构建类似VGG网络架构，Conv-BN-Conv-BN-Maxpool-Dropout<br>
  (2)采用MNIST数据集构建Densenet网络架构，自己可以直接把Densenet网络架构用于声学模型；<br>
  (3)为了解决CTC在最后Dense层分类无法捕捉前后信息，我对Dense层加入了Attention机制，代码如下：
        ####融入了attention机制于全连接层；<br>
        dense1 = Dense(units=512, activation='relu', use_bias=True, kernel_initializer='he_normal')(reshape)<br>
        attention_prob = Dense(units=512, activation='softmax', name='attention_vec')(dense1)<br>
        attention_mul = multiply([dense1, attention_prob])<br>
        dense1 = BatchNormalization(epsilon=0.0002)(attention_mul)<br>
        dense1 = Dropout(0.3)(dense1)<br>
## 实验结果
  DCNN-CTC:25.09%<br>
  Densenet-CTC:SER=31.23%<br>
  DCNN-Attention-CTC:SER=24.32%<br>
## 实验改进
  欢迎各位同行对本声学模型提供宝贵建议；
