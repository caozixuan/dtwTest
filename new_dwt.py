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
from dtwTest import get_predict_next_array_by_array,get_data,ge_cause,forward_shift,add_noise, normal, get_predict_array2, DistanceElement2

predict_length = 3
min_length = 3
max_length = 6
window = 20
compare_size = 4
title, title_copy = get_data()
g1 = ge_cause(200,10)
g2 = forward_shift(g1,5)
g2 = add_noise(5,g2)
es = ExponentialSmoothing(title[0])
model = es.fit()
array1 = model.predict(1, -1)
array1 = normal(array1)
es2 = ExponentialSmoothing(title[8])
model2 = es2.fit()
array2 = model2.predict(1, -1)
oil = normal(array2)
d = []
elements = []
errors = []
g1 = [10,8,7,9,11,13,10,9,11,13,15,11,14,16,18,19,20,22,25,26,28,27,29,30,31,35,34,32,30,31,28,26,23,20,15,14,10,9,10,11,8,7,5,9,10,12,13,14,10,13]
g2 = [10,12,11,9,8,9,10,11,12,10,9,11,11,12,10,9,8,10,12,14,15,17,19,22,23,25,28,27,30,34,35,36,39,37,33,30,28,27,26,24,20,16,15,14,12,11,8,9,10,11]
#g1 = ge_cause(100,5)
#g2 = forward_shift(g1,5)
g2 = add_noise(5,g2)
result, error_sum = get_predict_array2(array1, predict_length, 0)
all_errors = []
for z in range(0,compare_size+window):
    errors.append(0)
    result[z]=0
print array1
print oil
print len(result)
for i in range(compare_size+window, len(result) - predict_length):
    predict_array = array1[i - compare_size:i]
    use_array = oil[i - compare_size - window:i+predict_length]
    element = get_predict_next_array_by_array(predict_array, use_array, 0, 0,predict_length)
    elements.append(element)
    #all_errors.append(abs(element.next_element - array1[i]))
    if element.distance <= 5:
        #print element.distance
        #d.append(element.distance)
        #print element.array.tolist()
        #print predict_array
        #print element.next_element
        #print array1[i]
        #print '************************'
        next_array = np.array(element.next_array).reshape(-1,1)
        next_predict = np.array(array1[i:i+predict_length]).reshape(-1,1)
        dist, cost, acc, path = dtw(next_array, next_predict,
                                    dist=lambda next_array, next_predict: np.linalg.norm(
                                        next_array - next_predict, ord=1))
        errors.append(dist)
        print "*************"
        print element.array
        print predict_array
        print element.next_array
        print array1[i:i + predict_length]
        print "*************"
        """
        if dist<result[i]:
            print "*************"
            print element.array
            print predict_array
            print element.next_array
            print array1[i:i+predict_length]
            print "*************"
        """
    else:
        errors.append(result[i])
print len(result)
print len(errors)
#plt.plot(array1)
#plt.plot(oil)
print sum(errors)
print sum(result[0:len(result) - 1])
plt.plot(result[0:len(result) - 1])
plt.plot(errors)
#plt.plot(all_errors)
plt.show()

