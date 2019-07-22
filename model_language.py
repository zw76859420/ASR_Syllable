#-*- coding:UTF-8 -*-
#author:zhangwei

class ModelLanguage():
    def __init__(self , modelpath):
        self.modelpath = modelpath

        self.slash = '/'
        if self.slash != self.modelpath[-1]:
            self.modelpath = self.modelpath + self.slash
        pass

    def load_model(self):
        self.dict_pinyin = self.get_symbol_dict('dict.txt')
        self.model1 = self.get_language_model(self.modelpath + 'language_model1.txt')
        self.model2 = self.get_language_model(self.modelpath + 'language_model2.txt')
        self.pinyin = self.get_pinyin(self.modelpath + 'dic_pinyin.txt')
        model = (self.dict_pinyin , self.model1 , self.model2)
        return model

    def get_symbol_dict(self , dict_filename):
        dic_symbol = {}
        list_symbol = []
        with open(dict_filename , 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                res = line.split('\n')
                for i in res:
                    if i != '':
                        txt_1 = i.split('\t')
                        dic_symbol[txt_1[0]] = txt_1[1]
                        list_symbol.append(txt_1[0])
        return dic_symbol

    def get_language_model(self , modelname):
        dic_model = {}
        with open(modelname, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                res = line.split('\n')
                for i in res:
                    if i != '':
                        txt_1 = i.split('\t')
                        if len(txt_1) == 1:
                            continue
                        dic_model[txt_1[0]] = txt_1[1]
        return dic_model

    def get_pinyin(self , filename):
        dic = {}
        with open(filename, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                res = line.split('\n')
                for i in res:
                    if i == '':
                        continue
                    pinyin_split = i.split('\t')
                    list_pinyin = pinyin_split[0]
                    if (list_pinyin not in dic and int(pinyin_split[1]) > 1):
                        dic[list_pinyin] = pinyin_split[1]
        return dic

    def decode(self , list_syllabel , yuzhi=0.0001):
        list_words = []
        num_pinyin = len(list_syllabel)
        # print(num_pinyin)
        for i in range(num_pinyin):
            if list_syllabel[i] in self.dict_pinyin:
                ls = self.dict_pinyin[list_syllabel[i]]
                # print(ls)
            else:
                break
            if i == 0:
                num_ls = len(ls)
                # print(ls , num_ls)
                for j in range(num_ls):
                    # tuple_word = ['' , 0.0]
                    tuple_word = [ls[j] , 1.0]
                    list_words.append(tuple_word)
                # print(list_words)
                continue
            else:
                # print(list_words)
                list_words_2 = []
                num_ls_word = len(list_words)
                # print(num_ls_word)
                # print(ls)
                for j in range(0 , num_ls_word):
                    num_ls = len(ls)
                    # print(num_ls)
                    for k in range(0 , num_ls):
                        tuple_word = ['' , 0.0]
                        tuple_word = list(list_words[j])
                        # print(tuple_word[0])
                        # print(ls[k])
                        tuple_word[0] = tuple_word[0] + ls[k]
                        # print(tuple_word[0])
                        tmp_words = tuple_word[0][-2:]
                        # print(tmp_words)
                        if tmp_words in self.model2:
                            # print(tmp_words , tmp_words in self.model2)
                            tuple_word[1] = tuple_word[1] * float(self.model2[tmp_words]) / float(self.model1[tmp_words[-2]])
                            #print(self.model2[tmp_words] , self.model1[tmp_words[-2]])
                            #print(tuple_word[1])
                        else:
                            tuple_word[1] = 0.0
                            continue
                        # print(tuple_word)
                        # print(tuple_word[1] >= pow(yuzhi , 1))
                        if tuple_word[1] >= pow(yuzhi , i):
                            list_words_2.append(tuple_word)
                list_words = list_words_2
        # print(list_words)

        for i in range(0 , len(list_words)):
            # print(i)
            for j in range(i + 1 , len(list_words)):
                if list_words[i][1] < list_words[j][1]:
                    tmp = list_words[i]
                    list_words[i] = list_words[j]
                    list_words[j] = tmp
        return list_words

    def speech_to_text(self , list_syllabel):
        r = ''
        length = len(list_syllabel)
        if length == 0:
            return ''
        str_tmp = [list_syllabel[0]]

        for i in range(0 , length - 1):
            str_split = list_syllabel[i] + ' ' +list_syllabel[i + 1]
            if str_split in self.pinyin:
                str_tmp.append(list_syllabel[i + 1])
            else:
                str_decode = self.decode(str_tmp , 0.0000)
                # print(str_tmp , str_decode)
                if str_decode != []:
                    r += str_decode[0][0]
                str_tmp = [list_syllabel[i + 1]]
        str_decode = self.decode(str_tmp , 0.0000)

        if str_decode != []:
            r += str_decode[0][0]
        return r


if __name__ == '__main__':
    modelpath = '/home/zhangwei/PycharmProjects/ASR_Thchs30/model_language/'
    ms = ModelLanguage(modelpath=modelpath)
    ms.load_model()
    list_syllabel = ['wu2' , 'xi1']
    r = ms.speech_to_text(list_syllabel)
    print(r)




