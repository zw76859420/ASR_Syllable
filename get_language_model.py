#-*- coding:utf-8 -*-
#author:zhangwei

def get_language_model(filename):
    dic_model = {}
    with open(filename , 'r' , encoding='UTF-8') as fr:
        lines = fr.readlines()
        for line in lines:
            res = line.split('\n')
            for i in res:
                if(i != ''):
                    txt_1 = i.split('\t')
                    dic_model[txt_1[0]] = txt_1[1]
    print(dic_model)

if __name__ == '__main__':
    filename = 'F:\\asr_lm\\model_language\\language_model1.txt'
    get_language_model(filename)