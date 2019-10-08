# -*- coding: utf-8 -*-
# @Time    : 2019/9/16 20:10
# @Author  : QixianZhou
# @FileName: LocationEstimationDemo06.py
# @Abstract: 位置预测算法 Demo2 少量数据 采用numpy存储的网格  结果也采用numpy存储
# 20190917 基于600m网格时间精度为 秒 预测数据
# 修改tdoa算法后的测试类
# 伪造一条特定的数据 计算它的tdoa然后做计算


import numpy as np
import pandas as pd
import json
import pickle
import math
# 监控区域到地面的距离 11582
HEIGHT = 11582
# 简单的网格计算 500m * 500m
WIDTH = 150 * 1000
LENGTH = 220 * 1000
SQURE_SIZE = 600
# 光速 m/s
C = 299792458

# 三个sensor的信息 也初始化为常量
# 三个传感器的序列号 分别为 322 394 436
# 所以这里tdoa的顺序是
sensor_A_lat = 51.833122
sensor_A_lon = 4.142814
sensor_A_height = -1.820427

sensor_B_lat = 51.242813
sensor_B_lon = 6.684991
sensor_B_height = 43.901066

sensor_C_lat = 51.341648
sensor_C_lon = 5.893123
sensor_C_height = 39.085175
# 飞机的tdoa需要按照这个顺序生成
sensors = [322,394,436]
# 秒到纳秒的换算
S2NS = 1000000000


def calcTDOAByMessage(list):
    '''
    根据message中的数据计算tdoa 时间s
    :param list: 
    :return: 
    '''

    list[1] = (list[1]/S2NS - list[0]/S2NS)
    list[2] = (list[2]/S2NS - list[0]/S2NS)
    list[0] = 0
    return np.array(list)




def calcTopKInex(tdoa,grid_numpy):
    # print("---------")
    # print(grid_numpy.shape[0])


    # 先计算tdoa和grid_numpy的每一条记录的欧式距离
    distance = np.zeros(grid_numpy.shape[0])
    for i in range(distance.shape[0]):
        distance[i] = np.linalg.norm(tdoa - grid_numpy[i])
    # 计算topk 并返回索引
    top_k_index = np.argpartition(distance, 10)

    return top_k_index[0:10]





def calcLocationEstimation(top_k_index,grid):
    # print("topk")
    # print(top_k_index)
    # 依次获取位置
    sum_lat = 0
    sum_lon = 0
    for i in range(len(top_k_index)):
        sum_lat += grid[top_k_index[i]][1]
        sum_lon += grid[top_k_index[i]][2]

    return sum_lat/len(top_k_index),sum_lon/len(top_k_index)



from geopy.distance import geodesic




def calcDistanceContainHeight(point_distance,sensor_height):
    '''
    计算含高度信息的距离
    :param point_distance: 两点之间的距离
    :param sensor_height: sensor的高度
    :return: 
    '''
    height = HEIGHT - sensor_height
    sum = point_distance**2 + height**2
    return math.sqrt(sum)


def calcTime(lat,lon,sensor_lat,sensor_lon,sensor_height):
    '''
    计算一个位置到一个sensor的时间
    :param lat: 
    :param lon: 
    :param sensor_lat: 
    :param sensor_lon: 
    :param sensor_height: 
    :return: 
    '''
    # 先计算两点的距离
    distance = geodesic((lat, lon), ( sensor_lat ,  sensor_lon )).m
    # 再计算加上高度的距离
    distance = calcDistanceContainHeight(distance,sensor_height)
    # 把km->m
    # 单位 秒
    time = distance / C
    return time

def calcTDOA(lat, lon):
    list = []
    # 先计算当前网格位置离三个sensor的时间 (单位 纳秒)
    # 以A的时间为基准
    t_A = calcTime(lat,lon,sensor_A_lat,sensor_A_lon,sensor_A_height)
    t_B = calcTime(lat,lon,sensor_B_lat,sensor_B_lon,sensor_B_height)
    t_C = calcTime(lat,lon,sensor_C_lat,sensor_C_lon,sensor_C_height)
    list.append(t_A)
    list.append(t_B)
    list.append(t_C)
    #以其中某一个为基准后 就不需要排序了
    list[1] = list[1] - list[0]
    list[2] = list[2] - list[0]
    list[0] = 0
    # 转换为array返回
    return np.array(list)




# 600m在纬度上的距离
deleta_lon = (4.142814-4.134092)
deleta_lat = (51.833122-51.827732)
if __name__ == '__main__':
    # 先读取网格数据
    grid = np.load("Grid_600m_s.npy")


    grid_numpy = grid[:,3:6]
    print(grid_numpy.shape)


    # 自定义一条数据 计算位置和tdoa
    lat = sensor_A_lat - 20 * deleta_lat + deleta_lat/2
    lon = sensor_A_lon + 20 * deleta_lon + deleta_lon/2
    # 计算它的tdoa
    tdoa = calcTDOA(lat,lon)
    print("待预测数据的位置为",lat,lon,"tdoa为",tdoa)
    top_k_index = calcTopKInex(tdoa, grid_numpy)

    print("******************************************")
    print("当前飞机的位置和tdoa为：", lat, lon, tdoa)
    print("******************************************")
    print("打印输出预测出的top5个网格,以及这5个网格的相关信息")
    for x in top_k_index:
        print("当前网格的序号为：", x)
        print("当前网格的位置和tdoa为", grid[x][1:3], grid[x][3:6])
        tdoa_eula = np.linalg.norm(grid_numpy[x] - tdoa)
        print("当前网格的tdoa和飞机的tdoa的欧式距离为：", tdoa_eula)
        error = geodesic((lat, lon), (grid[x][1], grid[x][2])).m
        print("当前网格与飞机的真实位置距离为%.2fm" % (error))
        print("************************************************")

    # 根据索引，计算位置的预测值
    predictLatitude,predictLongitude = calcLocationEstimation(top_k_index,grid)
    print("打印预测位置")
    print(predictLatitude,predictLongitude)

    # 计算误差
    error = geodesic((lat, lon), ( predictLatitude ,  predictLongitude )).m
    print("打印预测误差")
    print(error)









