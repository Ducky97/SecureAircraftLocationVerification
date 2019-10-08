# -*- coding: utf-8 -*-
# @Time    : 2019/9/16 20:10
# @Author  : QixianZhou
# @FileName: LocationEstimationDemo06.py
# @Abstract: 位置预测算法 Demo2 少量数据 采用numpy存储的网格  结果也采用numpy存储
# 20190917 基于600m网格时间精度为 秒 预测数据
# 修改tdoa算法后的测试类
# 尝试不用飞机的时间数据来生成tdoa，而利用它的真实位置信息来生成tdoa

import numpy as np
import pandas as pd
import json
import pickle
import math

# 飞机的tdoa需要按照这个顺序生成
sensors = [322,394,436]
# 秒到纳秒的换算
S2NS = 1000000000
HEIGHT = 11 * 1000
C = 299792458

def calcTDOAByMessage(list):
    '''
    根据message中的数据计算tdoa 时间s
    :param list: 
    :return: 
    '''

    list[1] = (list[1] - list[0])/S2NS
    list[2] = (list[2] - list[0])/S2NS
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
    top_k_index = np.argpartition(distance, 5)
    return top_k_index[0:5]

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


sensor_A_lat = 51.833122
sensor_A_lon = 4.142814
sensor_A_height = -1.820427

sensor_B_lat = 51.242813
sensor_B_lon = 6.684991
sensor_B_height = 43.901066

sensor_C_lat = 51.341648
sensor_C_lon = 5.893123
sensor_C_height = 39.085175

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
    # return np.array(list)
    return list

if __name__ == '__main__':
    # 先读取网格数据
    grid = np.load("Grid_600m_s.npy")


    grid_numpy = grid[:,3:6]
    print(grid_numpy.shape)


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
        print("打印真实位置")
        print(lat,lon)
        j = json.loads(row['measurements'])
        # 这里要注意 我们这里需要判断精确的 按照熟322 394 436的顺序找到三个传感器
        # 所以这里需要一个遍历
        # 同时用一个list来接收数据
        list = [0,0,0]
        sensors_serial = [0,0,0]
        for i in range(len(j)):
            if sensors.__contains__(j[i][0]):
                if j[i][0] == 322:
                    sensors_serial[0] = j[i][0]
                    list[0] = j[i][1]
                if j[i][0] == 394:
                    sensors_serial[1] = j[i][0]
                    list[1] = j[i][1]
                if j[i][0] == 436:
                    sensors_serial[2] = j[i][0]
                    list[2] = j[i][1]

        print("查看飞机的tdoa前的数据")
        print(sensors_serial)
        print(list)
        # print("查看挑选的list，是不是对应的sensor的数据")
        # print(list)
        # 计算tdoa
        tdoa_time = calcTDOAByMessage(list)
        print("查看通过时间计算出来的tdoa",tdoa_time)

        print("尝试通过位置来生成tdoa")
        tdoa_position = calcTDOA(lat,lon)
        tdoa_position = np.array(tdoa_position)
        print("查看通过位置计算出来的tdoa", tdoa_position)


        # 拿到tdoa后 可以开始预测位置了
        # 这里返回的是topk的索引 然后根据索引再去grid表中找对应的位置信息 再求平均值
        top_k_index = calcTopKInex(tdoa_position,grid_numpy)
        print("打印topk索引")
        print(top_k_index)
        #打印输出最近的五个点
        print("打印输出预测的五个点")
        for x in top_k_index:
            print(grid[x][1:3])



        # 根据索引，计算位置的预测值
        predictLatitude,predictLongitude = calcLocationEstimation(top_k_index,grid)
        print("打印预测位置")
        print(predictLatitude,predictLongitude)

        # 计算误差
        error = geodesic((lat, lon), ( predictLatitude ,  predictLongitude )).m
        print("打印预测误差")
        print(error)
        # 保存数据
        predictError =  predictError.append([{'id': id, 'latitude': lat, "longitude": lon, 'predictLatitude': predictLatitude, 'predictLongitude': predictLongitude,
                             'error': error}])

        count+=1
        # print("完成%d条消息预测"%count)
        # # 测试100条数据
        print("-----------------------------------")
        if count >= 5:
            break




    # 保存数据
    # predictError.to_csv("predictError_600m_s.csv")
    # print("打印误差的平均值")
    # mean = predictError['error'].mean()
    # median = predictError['error'].median()
    # print("基于600m网格和ns精度，完成%d条消息的位置预测，平均误差为:%.2f,中位数为:%.2f" % (count, mean,median))
