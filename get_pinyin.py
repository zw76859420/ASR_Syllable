#-*- coding:utf-8 -*-
#author:zhangwei

filename = 'F:\\asr_lm\\model_language\\dic_pinyin.txt'
def get_pinyin(filename):
    dic = {}
    with open(filename , 'r' , encoding='UTF-8') as fr:
        lines = fr.readlines()
        for line in lines:
            res = line.split('\n')
            for i in res:
                if i == '':
                    continue
                pinyin_split = i.split('\t')
                list_pinyin = pinyin_split[0]
                if(list_pinyin not in dic and int(pinyin_split[1]) > 1):
                    dic[list_pinyin] = pinyin_split[1]
        print(dic)


