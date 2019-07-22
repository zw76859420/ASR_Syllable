# -*- coding:utf-8 -*-
# author:zhangwei

class ModelLanguage():
    def __init__(self, modelpath):
        self.modelpath = modelpath
        self.slash = '/'
        if (self.slash != self.modelpath[-1]):
            self.modelpath = self.modelpath + self.slash
        pass

    def LoadModel(self):
        self.dict_pinyin = self.GetSymbolDict('dict.txt')
        self.pinyin = self.GetPinyin(self.modelpath + 'dic_pinyin.txt')
        self.model1 = self.GetLanguageModel(self.modelpath + 'language_model1.txt')
        self.model2 = self.GetLanguageModel(self.modelpath + 'language_model2.txt')
        model = (self.dict_pinyin, self.model1, self.model2)
        return model

    def GetSymbolDict(self, dictfilename):
        '''
        		读取拼音汉字的字典文件
        		返回读取后的字典
        		'''
        txt_obj = open(dictfilename, 'r', encoding='UTF-8')  # 打开文件并读入
        txt_text = txt_obj.read()
        txt_obj.close()
        txt_lines = txt_text.split('\n')  # 文本分割

        dic_symbol = {}  # 初始化符号字典
        for i in txt_lines:
            list_symbol = []  # 初始化符号列表
            if (i != ''):
                txt_l = i.split('\t')
                pinyin = txt_l[0]
                for word in txt_l[1]:
                    list_symbol.append(word)
            dic_symbol[pinyin] = list_symbol

        return dic_symbol

    def GetPinyin(self, filename):
        file_obj = open(filename, 'r', encoding='UTF-8')
        txt_all = file_obj.read()
        file_obj.close()

        txt_lines = txt_all.split('\n')
        dic = {}

        for line in txt_lines:
            if (line == ''):
                continue
            pinyin_split = line.split('\t')

            list_pinyin = pinyin_split[0]

            if (list_pinyin not in dic and int(pinyin_split[1]) > 1):
                dic[list_pinyin] = pinyin_split[1]
        return dic

    def GetLanguageModel(self, modelLanFilename):
        txt_obj = open(modelLanFilename, 'r', encoding='UTF-8')  # 打开文件并读入
        txt_text = txt_obj.read()
        txt_obj.close()
        txt_lines = txt_text.split('\n')  # 文本分割

        dic_model = {}  # 初始化符号字典
        for i in txt_lines:
            if (i != ''):
                txt_l = i.split('\t')
                if (len(txt_l) == 1):
                    continue
                # print(txt_l)
                dic_model[txt_l[0]] = txt_l[1]

        return dic_model

    def SpeechToText(self, list_syllable):
        r = ''
        length = len(list_syllable)
        str_tmp = [list_syllable[0]]
        for i in range(0, length-1):
            str_split = list_syllable[i] + ' ' + list_syllable[i+1]
            if (str_pinyin in self.pinyin):
                str_tmp.append(list_syllable[i+1])
            else:
                pass


    def decode(self, list_syllable, yuzhi=0.0001):
        list_words = []
        num_pinyin = len(list_syllable)
        for i in range(num_pinyin):
            ls = ''
            if (list_syllable[i] in self.dict_pinyin):
                ls = self.dict_pinyin[list_syllable[i]]
                # print(ls)
            else:
                break

            if (i == 0):
                num_ls = len(ls)
                for j in range(num_ls):
                    tuple_word = [ls[j], 1.0]
                    list_words.append(tuple_word)
                # print(list_words)
                continue

            else:
                list_words_2 = []
                num_ls_word = len(list_words)
                for j in range(0, num_ls_word):
                    num_ls = len(ls)
                    for k in range(0, num_ls):
                        tuple_word = list(list_words[j])
                        tuple_word[0] = tuple_word[0] + ls[k]
                        tmp_words = tuple_word[0][-2:]
                        print(tmp_words)

if __name__ == '__main__':
    ml = ModelLanguage(modelpath='model_language')
    ml.LoadModel()
    str_pinyin = ['kao3', 'yan2', 'yan1', 'yu3', 'ci2', 'hui4']
    # ml.SpeechToText(list_syllable=str_pinyin)
    ml.decode(list_syllable=str_pinyin)