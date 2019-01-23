# ASR_Syllable
  采用端到端方法构建声学模型，以字为建模单元，采用DCNN-CTC网络结构，希望集众人力量改进以音节为建模单元的声学模型。
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
  采用DCNN-CTC构建语音识别声学模型；<br>
  CTC损失函数介绍可以观看本人的博客；<br>
  [CTC原理介绍]（https://blog.csdn.net/Xwei1226/article/details/80969250 "悬停显示"）;<br>
  [CTC损失函数推到](https://blog.csdn.net/Xwei1226/article/details/80889818 "悬停显示");<br>
