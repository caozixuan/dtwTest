# encoding:utf-8
from dtw import dtw
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import csv
import random
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def chafen(a):
    result = []
    for i in range(1, len(a)):
        result.append(a[i] - a[i - 1])
    return result


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
    def __init__(self, array, distance, next_element=0):
        self.array = array
        self.distance = distance
        self.next_element = next_element


# 保存相似数组和距离
class DistanceElement2:
    def __init__(self, array, distance, next_array):
        self.array = array
        self.distance = distance
        self.next_array = next_array

min_length = 3
max_length = 6
window = 20
compare_size = 4
"""
在一定范围内寻找两个数组之间最相似的序列,从大到小开始找，目前定义匹配的数组最长为6，
最短为3
"""


def get_predict_distance(a, b, b_long):
    distances = [DistanceElement([0, 0, 0, 0, 0], 1000)]
    i = max_length
    while i >= min_length and len(a) - i >= 0:
        array1_average = 0.0
        compare_array1 = a[len(a) - i: len(a)]
        for l in range(0, len(compare_array1)):
            array1_average += compare_array1[l][0]
        array1_average = array1_average / len(compare_array1)
        j = 6
        while j >= 3 and len(b) - j >= 0:
            k = 0
            while k <= window - max_length and len(b) - j - k >= 0:
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

    # 返回距离最小的元素（最相似）
    return distances[0]


def get_predict_next_by_array(predict_array, use_array, co_length, next_element):
    i = max_length
    distances = []
    while i >= min_length:
        j = 0
        while j <= window - max_length:
            simi_array = use_array[len(use_array) - i - j - 1:len(use_array) - j - 1]
            #simi_array = normal(simi_array)
            #predict_array = normal(predict_array)
            trend_use_array = (use_array[len(use_array) - j - 1]-simi_array[-1])/(simi_array[-1]*1.0)+1
            co = (sum(simi_array)*1.0/len(simi_array)/(sum(predict_array)*1.0/len(predict_array)))
            co_invert = 1.0/co
            #simi_array = co*np.array(predict_array)
            simi_array = np.array(simi_array).reshape(-1, 1)
            predict_array_tmp = np.array(predict_array).reshape(-1, 1)
            dist, cost, acc, path = dtw(simi_array, predict_array_tmp,
                                        dist=lambda simi_array, predict_array_tmp: np.linalg.norm(
                                            simi_array - predict_array_tmp, ord=1))
            #distances.append(DistanceElement(simi_array, dist, co_invert*use_array[len(use_array) - j - 1]))
            distances.append(DistanceElement(simi_array, dist/i, trend_use_array*predict_array[-1]))
            if dist==0:
                print 'why is zero'
            j += 1
        i -= 1
    distances.sort(cmp=None, key=lambda x: x.distance, reverse=False)
    return distances[0]


def get_predict_next_array_by_array(predict_array, use_array, co_length, next_element,length):
    i = max_length
    distances = []
    while i >= min_length:
        j = 0
        while j <= window - max_length:
            simi_array = use_array[len(use_array) - i - j-length:len(use_array) - j-length]
            co = (sum(simi_array)*1.0/len(simi_array)/(sum(predict_array)*1.0/len(predict_array)))
            co_invert = 1.0/co
            #simi_array = co*np.array(predict_array)
            simi_array = np.array(simi_array).reshape(-1, 1)
            predict_array_tmp = np.array(predict_array).reshape(-1, 1)
            dist, cost, acc, path = dtw(simi_array, predict_array_tmp,
                                        dist=lambda simi_array, predict_array_tmp: np.linalg.norm(
                                            simi_array - predict_array_tmp, ord=1))
            distances.append(DistanceElement2(simi_array, dist, co_invert*np.array(use_array[len(use_array) - j - length:len(use_array) - j])))
            if dist==0:
                print 'why is zero'
            j += 1
        i -= 1
    distances.sort(cmp=None, key=lambda x: x.distance, reverse=False)
    return distances[0]

import math


# 把所有的取值归为0-100
def normal(a):
    a = map(float, a)
    min_value = min(a)
    max_value = max(a)
    if max_value==min_value:
        return a
    for i in range(0, len(a)):
        a[i] = (a[i] - min_value) / (max_value - min_value)
        if a[i]==0:
            a[i]=0.0001
        #a[i] = a[i] * 100 + 1
        #if a[i] - int(a[i]) >= 0.5:
        #    a[i] = int(a[i]) + 1
        #else:
        #    a[i] = int(a[i])
        #a[i] = math.log(a[i])
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



"""
test1 = [10,12,9,8,10,11,12,10,10,9,9,11,11,13,15,17,19,25,20,18,17,13,11,12,10,9,10,10,9,9,11,10,11,11,10]
test2 = [40,41,43,39,41,42,43,40,39,39,40,41,42,40,39,38,39,45,50,55,60,61,55,54,50,43,42,42,41,40,39,38,38,40,40]
test1 = map(float, test1)
test2 = map(float, test2)
a = title[8]  # title[4]#title[3]#m_price#ge_cause(200, 10)
#a = chafen(a)
# a = title[8]
a = normal(a)
b = title[7]  # title[0]#title[0]#move#forward_shift(a, 5)
# b = add_noise(5,b)
#b = chafen(b)
b = normal(b)
print a
print b
"""
"""
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
        distances.append(min(x,y))
        distances_long.append(y)
plt.plot(distances)
plt.plot(distances_long)
plt.show()
# x = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
# y = np.array([1,1,2,3,4,5,6,7,8,9]).reshape(-1,1)
# print get_predict_distance(x,y)
"""


def ar_predict(input_array, lag, difference=0):
    input_array = map(float, input_array)
    model = ARMA(input_array, order=[lag, difference])
    result_arma = model.fit(disp=-1, method='css')
    predict_array = result_arma.predict()
    real_array = input_array[lag:]
    difference_array = []
    for j in range(0, lag):
        difference_array.append(0)
    for i in range(0, len(predict_array)):
        difference_array.append(abs(predict_array[i] - real_array[i]))
    return difference_array


def ar_predict_next_element(input_array, lag, difference=0):
    input_array = map(float, input_array)
    model = ARMA(input_array, order=[lag, difference])
    result_arma = model.fit(disp=-1, method='mle')
    predict_array = result_arma.predict(len(input_array), len(input_array))
    return predict_array[len(predict_array) - 1]


def ar_predict_next_array(input_array, lag, length, difference=0):
    input_array = map(float, input_array)
    model = ARMA(input_array, order=[lag, difference])
    result_arma = model.fit(disp=-1, method='mle')
    predict_array = result_arma.predict(len(input_array), len(input_array)+length)
    return predict_array

def get_predict_array(whole_array, lag, difference=1):
    result = []
    error_sum = []
    for i in range(0, len(whole_array) - 1):
        if i < 10:
            result.append(0)
            error_sum.append(0)
        else:
            error = abs(ar_predict_next_element(whole_array[i-10:i], lag, difference) - whole_array[i])
            result.append(error)
            error_sum.append(error + error_sum[i - 1])
    return result, error_sum

predict_length = 3
def get_predict_array2(whole_array, lag, difference=0):
    result = []
    error_sum = []
    for i in range(0, len(whole_array) - 1):
        if i < 10:
            result.append(0)
            error_sum.append(0)
        else:
            predict_array = np.array(ar_predict_next_array(whole_array[i-10:i], lag,predict_length,difference)).reshape(-1,1)
            next_array = np.array(whole_array[i:i+predict_length]).reshape(-1,1)
            dist, cost, acc, path = dtw(predict_array, next_array,
                                        dist=lambda predict_array, next_array: np.linalg.norm(
                                            predict_array - next_array, ord=1))
            result.append(dist)
            error_sum.append(dist + error_sum[i - 1])
    return result, error_sum

def change_to_continue(a):
    result = []
    base = 1
    result.append(base * (1 + a[0]))
    for i in range(1, len(a)):
        result.append((1 + a[i]) * result[i - 1])
    return result


def difference_predict(input_array):
    # 获取要比较的数据

    es = ExponentialSmoothing(input_array)
    model = es.fit()
    predict_array = model.predict()
    print input_array
    print predict_array
    difference_array = []
    for i in range(1, len(predict_array)):
        difference_array.append(abs(predict_array[i] - input_array[i - 1]))
    return difference_array


"""
input_array1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
input_array2 = [1,2,3,4,5,4,3,2,1,2,3,4,5,4,3,2,1,2,3,4,5,4,3,2,1]
input_array3 = [1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,7,3,2,1,2,3,4,5,6,7,8,9,10]
input_array4 = title[8]
input_array5 = title[0]
input_array6 = [40,41,43,39,41,42,43,40,39,39,40,41,42,40,39,38,39,45,50,55,60,61,55,54,50,43,42,42,41,40,39,38,38,40,40]
input_array7 = [20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
plt.plot(normal(input_array5))
print title[8]
difference_array,error_sum = get_predict_array(title[0],3)
plt.plot(normal(difference_array))
print error_sum
plt.plot(normal(error_sum))
plt.show()

es = ExponentialSmoothing(title[0])
model = es.fit()
array1 = model.predict(1,-1)
es2 = ExponentialSmoothing(title[8])
model2 = es2.fit()
array2 = model2.predict(0,-1)
print array1
print title[0]
plt.plot(normal(array1))
plt.plot(normal(title[0]))
print len(array1)
print len(title[0])
plt.show()
"""
if __name__ == '__main__':
    move, r_price, m_price = get_MR_data()
    title, title_copy = get_data()
    trend, usd = read_from_csv()
    trend = map(float, trend)
    usd = map(float, usd)
    g1 = ge_cause(100, 5)
    g2 = forward_shift(g1, 5)
    g2 = add_noise(5, g2)
    g3 = ge_cause(100, 10)
    es = ExponentialSmoothing(title[0])
    model = es.fit()
    array1 = model.predict(1, -1)
    array1 = normal(array1)
    print max(title[4])
    print min(title[4])
    es2 = ExponentialSmoothing(title[5])
    model2 = es2.fit()
    array2 = model2.predict(1, -1)
    oil = normal(array2)
    d = []
    elements = []
    errors = []
    array1 = g1
    oil = g3
    result, error_sum = get_predict_array(array1, 3, 0)
    all_errors = []
    for z in range(0, compare_size + window):
        errors.append(0)
        result[z] = 0
    print array1
    print oil
    print len(result)
    better_points = []
    cause_points = []
    for i in range(compare_size + window, len(result) - 2):
        predict_array = array1[i - compare_size:i]
        use_array = oil[i - compare_size - window:i + 1]
        element = get_predict_next_by_array(predict_array, use_array, 0, 0)
        elements.append(element)
        all_errors.append(abs(element.next_element - array1[i]))
        if element.distance <= 1:
            errors.append(abs(element.next_element - array1[i]))
            if abs(element.next_element - array1[i]) < result[i]:
                print "*************"
                print element.array
                print predict_array
                print element.next_element
                print array1[i]
                print "*************"
                better_points.extend(predict_array)
                #better_points.append(array1[i])
                for x in element.array:
                    cause_points.append(x[0])
        else:
            errors.append(result[i])
    print len(result)
    print len(errors)
    plt.plot(array1)
    plt.plot(oil)
    print sum(errors)
    print sum(result[0:len(result) - 1])
    #plt.plot(result[0:len(result) - 1])
    #plt.plot(errors)
    # plt.plot(all_errors)
    fig = plt.figure()
    axes = fig.add_subplot(111)
    for i in range(len(array1)):
        if array1[i] in better_points:
            #  第i行数据，及returnMat[i:,0]及矩阵的切片意思是:i：i+1代表第i行数据,0代表第1列数据
            axes.scatter(i, array1[i], color='red')
        else:
            axes.scatter(i, array1[i], color='black')
        if oil[i] in cause_points:
            axes.scatter(i, oil[i], color='green')
        else:
            axes.scatter(i, oil[i], color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


