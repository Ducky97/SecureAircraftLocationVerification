# -*- coding: utf-8 -*-
# @Time    : 2019/9/16 20:10
# @Author  : QixianZhou
# @FileName: LocationEstimationDemo01.py
# @Abstract: 位置预测算法 Demo2 少量数据 采用numpy存储的网格  结果也采用numpy存储
# 20190917 基于600m网格时间精度为s预测数据
# 基于高精度网格的预测算法


import numpy as np
import pandas as pd
import json
import pickle
from decimal import Decimal
from decimal import getcontext
getcontext().prec = 30

sensors = [322,394,436]
S2NS = 1000000000
# def calcTDOAByMessage(list):
#     '''
#     根据message中的数据计算tdoa 时间s
#     :param list:
#     :return:
#     '''
#     list.sort()
#     list[1] = (Decimal(list[1]/1000000000) - Decimal(list[0]/1000000000))
#     list[2] = (Decimal(list[2]/1000000000) - Decimal(list[0]/1000000000))
#     list[0] = Decimal(0)
#     return np.array(list)
def calcTDOAByMessage(list):
    '''
    根据message中的数据计算tdoa 时间s
    :param list: 
    :return: 
    '''

    list[1] = Decimal((list[1]- list[0])/S2NS)
    list[2] = Decimal((list[2] - list[0])/S2NS)
    list[0] = Decimal(0)
    return np.array(list)


def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]




def calcTopKInex(tdoa,grid_numpy):
    # print("---------")
    # print(grid_numpy.shape[0])


    # 先计算tdoa和grid_numpy的每一条记录的欧式距离
    distance = np.zeros(grid_numpy.shape[0])
    for i in range(distance.shape[0]):
        distance[i] = Decimal(np.linalg.norm(tdoa - grid_numpy[i]))
    # 计算topk 并返回索引
    top_k_index = np.argpartition(distance, 5)
    return top_k_index[0:5]

def calcLocationEstimation(top_k_index,grid):
    print("topk")
    print(top_k_index)
    # 依次获取位置
    sum_lat = Decimal(0)
    sum_lon = Decimal(0)
    for i in range(len(top_k_index)):
        sum_lat += Decimal(grid[top_k_index[i]][1])
        sum_lon += Decimal(grid[top_k_index[i]][2])

    return Decimal(sum_lat/len(top_k_index)),Decimal(sum_lon/len(top_k_index))



from geopy.distance import geodesic









if __name__ == '__main__':
    # 先读取网格数据
    grid = np.load("Grid_600m_s_demical.npy")
    # print(grid.info())

    # 从网格数据中分离tdoa数据

    grid_numpy = grid[:,3:6]
    print(grid_numpy.shape)
    # print("网格个数：",len(grid))
    #
    # print(grid.info())
    # # 建立网格的numpy数组 便于后续的预测
    # grid_numpy = np.zeros((len(grid),3))
    # # 遍历网格
    # for index, row in grid.iterrows():
    #     # 获取三个数据
    #     temp = []
    #     temp.append(row['t0'])
    #     temp.append(row['t1'])
    #     temp.append(row['t2'])
    #     grid_numpy[index] = np.copy(np.array(temp))


    # print(grid_numpy.shape)


    # 读入飞机位置数据
    messages = pd.read_csv("MessagesReceivedByRadarcapeBy3_11w_322_394_436_MonitorArea0.csv")
    # print(messages.info())
    print("消息条数:",len(messages))

    # 新建dataframe 记录消息的预测情况
    # id是原消息的id 真实位置 预测位置 误差
    predictError = pd.DataFrame(columns=['id','latitude','longitude','predictLatitude','predictLongitude','error'])

    count = 0
    for index,row in messages.iterrows():
        # 获取当前消息的经纬度
        # print("当前消息")
        # print(row)
        id = row['id']
        lat = row['latitude']
        lon = row['longitude']
        # 获取当前消息的接收数据
        j = json.loads(row['measurements'])
        # 这里要注意 我们这里需要判断传感器类型 是否为那三个
        # 所以这里需要一个遍历
        # 同时用一个list来接收数据
        list = []
        for i in range(len(j)):
            if sensors.__contains__(j[i][0]):
                list.append(j[i][1])
        # print("查看挑选的list，是不是对应的sensor的数据")
        # print(list)
        # 计算tdoa
        tdoa = calcTDOAByMessage(list)
        # 拿到tdoa后 可以开始预测位置了
        # 这里返回的是topk的索引 然后根据索引再去grid表中找对应的位置信息 再求平均值
        top_k_index = calcTopKInex(tdoa,grid_numpy)
        # 根据索引，计算位置的预测值
        predictLatitude,predictLongitude = calcLocationEstimation(top_k_index,grid)
        # 计算误差
        error = Decimal(geodesic((lat, lon), ( predictLatitude ,  predictLongitude )).m)
        # 保存数据
        predictError =  predictError.append([{'id': id, 'latitude': lat, "longitude": lon, 'predictLatitude': predictLatitude, 'predictLongitude': predictLongitude,
                             'error': error}])

        count+=1
        print("完成%d条消息预测"%count)
        # 测试100条数据
        if count >= 5:
            break


    print("打印误差的平均值")
    mean = predictError['error'].mean()
    median = predictError['error'].median()
    print("基于600m网格和ns精度，完成%d条消息的位置预测，平均误差为:%.2f,中位数为:%.2f" % (count, mean, median))
    #
    # # 保存数据
    # predictError.to_csv("predictError_600m_s_demical.csv")

