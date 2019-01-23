#-*- coding:utf-8 -*-
#author:zhangwei

'''
这个函数是为评价预测的准确率做准备，采用的编辑距离计算预测值和真实值之间的距离
'''

import difflib

def get_edit_distance(str1 , str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None , str1 , str2)
    for tag , i1 , i2 , j1 , j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2 - i1 , j2 - j1)
        elif tag =='insert':
            leven_cost += j2 - j1
        elif tag == 'delete':
            leven_cost += i2 - i1
    return leven_cost


if __name__ == '__main__':
    print(get_edit_distance('ABCD' , 'DBFG'))
