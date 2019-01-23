#-*- coding:utf-8 -*-
#author:zhangwei

'''
此函数是用作于加载字典里面的符号，用于声学模型的训练以及语言模型的训练；
'''

def get_list_symbol(datapath):
    list_symbol = []

    with open('dict.txt' , 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            res = line.split()
            list_symbol.append(res[0])
    list_symbol.append('_')
    return list_symbol

if __name__ == '__main__':
    datapath = '/home/zhangwei/PycharmProjects/ASR_Thchs30/data_list/'
    print(get_list_symbol(datapath))

