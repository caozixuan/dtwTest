# encoding:utf-8
from dtw import dtw
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import csv
import random

from statsmodels.tsa.stattools import grangercausalitytests


# 这个函数从表格中读取数据，没有用到
def read_from_excel():
    ExcelFile = xlrd.open_workbook(r'data/01.xlsx')
    sheet = ExcelFile.sheet_by_name('Sheet1')
    cols0 = sheet.col_values(0)
    cols1 = sheet.col_values(1)
    return cols0, cols1


# 从csv中读取数据，读取的是2016年比特币流行指数trend与比特币价格两个序列
def read_from_csv():
    trend = []
    usd = []
    csvfile = open('data/2016_weekly_trends.csv', 'rU')
    reader = csv.DictReader(csvfile)
    for row in reader:
        trend.append(row['trend'])
        usd.append(row['usd'])
    csvfile.close()
    return trend, usd


# 读取的是近十年加拿大的经济各个指标数据
def get_data():
    title = ['GDP_monthly']
    title.append("house_price_index_monthly")
    title.append("unemployment_rate")
    title.append("CPI_core_monthly")
    title.append("CPI_monthly")
    title.append("trade_monthly")
    title.append("sell")
    title.append("new house")
    title.append("brent")
    title.append("WTI")
    # title.append("dubai")
    title_copy = list(title)

    with open("data/gdpLongReal.dat", "r") as reader:
        for i in range(len(title)):
            title[i] = map(float, reader.readline().rstrip().split())
    print max(title[0])
    print min(title[0])
    return title, title_copy


# 保存相似数组和距离
class DistanceElement:
    def __init__(self, array, distance):
        self.array = array
        self.distance = distance


"""
在一定范围内寻找两个数组之间最相似的序列,从大到小开始找，目前定义匹配的数组最长为6，
最短为3
"""
def get_predict_distance(a, b, b_long):
    distances = [DistanceElement([0, 0, 0, 0, 0], 1000)]
    i = 6
    while i >= 3 and len(a) - i >= 0:
        array1_average = 0.0
        compare_array1 = a[len(a) - i: len(a)]
        for l in range(0, len(compare_array1)):
            array1_average += compare_array1[l][0]
        array1_average = array1_average / len(compare_array1)
        j = 6
        while j >= 3 and len(b) - j >= 0:
            k = 0
            while k <= 4 and len(b) - j - k >= 0:
                compare_array2 = b[len(b) - j - k: len(b) - k]
                array2_average = 0.0
                for m in range(0, len(compare_array2)):
                    array2_average += compare_array2[m][0]
                array2_average = array2_average / len(compare_array2)
                dist, cost, acc, path = dtw(compare_array1, compare_array2,
                                            dist=lambda compare_array1, compare_array2: np.linalg.norm(
                                                compare_array1 - compare_array2, ord=1))
                # print dist
                if array2_average == 0 or array1_average == 0:
                    array2_average = array1_average = 1
                # 为了避免要匹配的数组取值不在一个范围内，计算数组平均值，并将一个数据乘上其比值（这里的处理方法可能有问题）
                next_array = (abs(array1_average) / abs(array2_average)) * b_long[len(b) - k:len(b) - k + j]
                # next_array = b_long[len(b) - k:len(b) - k + j]
                distances.append(DistanceElement(next_array, dist))
                k += 1
            j = j - 1
        i = i - 1
    distances.sort(cmp=None, key=lambda x: x.distance, reverse=False)
    # for element in distances:
    #    print element.array, ":", element.distance

    #返回距离最小的元素（最相似）
    return distances[0]


# 把所有的取值归为0-100
def normal(a):
    min_value = min(a)
    max_value = max(a)
    for i in range(0, len(a)):
        a[i] = (a[i] - min_value) / (max_value - min_value)
        a[i] = a[i] * 100
        if a[i] - int(a[i]) >= 0.5:
            a[i] = int(a[i]) + 1
        else:
            a[i] = int(a[i])
    return a


# 比较两个数组之间的相似性，代码来自dtw官方实例
def predictSimilarity(targetArray, useArray):
    dist, cost, acc, path = dtw(targetArray, useArray,
                                dist=lambda targetArray, useArray: np.linalg.norm(targetArray - useArray, ord=1))
    return dist


# 随机生成一段序列，从50开始，随机增减。
def ge_cause(len, step):
    result = []
    result.append(50)
    for i in range(1, len):
        flag = random.randint(0, 1)
        if flag == 0:
            result.append(result[i - 1] - random.randint(0, step))
        else:
            result.append(result[i - 1] + random.randint(0, step))
    return result


# 此处获取的是零售价格和批发价格的关系
def get_MR_data():
    import csv
    with open('data/data.csv', 'rb') as csvfile:
        move = []
        r_price = []
        m_price = []
        reader = csv.DictReader(csvfile)
        for row in reader:
            move.append(row['Move'])
            r_price.append(row['RPRICE'])
            m_price.append(row['MPRICE'])
        move = map(float, move)
        r_price = map(float, r_price)
        m_price = map(float, m_price)
    return move, r_price, m_price


# 将输入的序列前移一定的位数
def forward_shift(seq, shift, invert=False):
    """seq: a binary sequence
    """
    lseq = len(seq)
    sseq = [None] * lseq
    for i in xrange(lseq):
        if i >= shift:
            sseq[i] = seq[i - shift]
        else:
            flag = random.randint(0, 1)
            if flag == 0:
                sseq[i] = seq[shift] - random.randint(0, 10)
            else:
                sseq[i] = seq[shift] + random.randint(0, 10)
    return sseq


# 加入噪声，每个元素随机扰动
def add_noise(noise, a):
    for i in range(0, len(a)):
        flag = random.randint(0, 1)
        if flag == 0:
            a[i] = a[i] - random.randint(0, noise)
        else:
            a[i] = a[i] + random.randint(0, noise)
    return a


# 获取要比较的数据
move, r_price, m_price = get_MR_data()
title, title_copy = get_data()
trend, usd = read_from_csv()
trend = map(float, trend)
usd = map(float, usd)
a = trend  # title[4]#title[3]#m_price#ge_cause(200, 10)
# a = title[8]
a = normal(a)
b = usd  # title[0]#title[0]#move#forward_shift(a, 5)
# b = add_noise(5,b)
b = normal(b)
print a
print b
# dwt包中对array类型的处理
a = np.array(a).reshape(-1, 1)
b = np.array(b).reshape(-1, 1)
distances = []
distances_long = []
for i in range(0, len(a)):
    x = a[0:i]
    y = b[0:i]
    element = get_predict_distance(y, x, a[0:i + 10])
    if i + len(element.array) < len(b) and i - len(element.array) >= 0:
        x = predictSimilarity(b[i:i + len(element.array)], element.array)
        y = predictSimilarity(b[i:i + len(element.array)], b[i - len(element.array):i])
        distances.append(x)
        distances_long.append(y)
plt.plot(distances)
plt.plot(distances_long)
plt.show()
# x = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
# y = np.array([1,1,2,3,4,5,6,7,8,9]).reshape(-1,1)
# print get_predict_distance(x,y)
