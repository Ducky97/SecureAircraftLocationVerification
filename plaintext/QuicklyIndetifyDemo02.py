# -*- coding: utf-8 -*-
# @Time    : 2019/9/26 19:11
# @Author  : QixianZhou
# @FileName: QuicklyIndetifyDemo02.py
# @Abstract: 快速定位算法 Demo02 设置不同的阈值进行判断




import numpy
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from geopy.distance import geodesic
from geopy.distance import lonlat
import geopy
import geopy.distance
import math
import time
from scipy.ndimage import map_coordinates


from geographiclib.geodesic import Geodesic

wgs84_geoid = numpy.array([[13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
                            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13],  # 90N
                           [3, 1, -2, -3, -3, -3, -1, 3, 1, 5, 9, 11, 19, 27, 31, 34, 33, 34, 33, 34, 28, 23, 17, 13, 9,
                            4, 4, 1, -2, -2, 0, 2, 3, 2, 1, 1],  # 80N
                           [2, 2, 1, -1, -3, -7, -14, -24, -27, -25, -19, 3, 24, 37, 47, 60, 61, 58, 51, 43, 29, 20, 12,
                            5, -2, -10, -14, -12, -10, -14, -12, -6, -2, 3, 6, 4],  # 70N
                           [2, 9, 17, 10, 13, 1, -14, -30, -39, -46, -42, -21, 6, 29, 49, 65, 60, 57, 47, 41, 21, 18,
                            14, 7, -3, -22, -29, -32, -32, -26, -15, -2, 13, 17, 19, 6],  # 60N
                           [-8, 8, 8, 1, -11, -19, -16, -18, -22, -35, -40, -26, -12, 24, 45, 63, 62, 59, 47, 48, 42,
                            28, 12, -10, -19, -33, -43, -42, -43, -29, -2, 17, 23, 22, 6, 2],  # 50N
                           [-12, -10, -13, -20, -31, -34, -21, -16, -26, -34, -33, -35, -26, 2, 33, 59, 52, 51, 52, 48,
                            35, 40, 33, -9, -28, -39, -48, -59, -50, -28, 3, 23, 37, 18, -1, -11],  # 40N
                           [-7, -5, -8, -15, -28, -40, -42, -29, -22, -26, -32, -51, -40, -17, 17, 31, 34, 44, 36, 28,
                            29, 17, 12, -20, -15, -40, -33, -34, -34, -28, 7, 29, 43, 20, 4, -6],  # 30N
                           [5, 10, 7, -7, -23, -39, -47, -34, -9, -10, -20, -45, -48, -32, -9, 17, 25, 31, 31, 26, 15,
                            6, 1, -29, -44, -61, -67, -59, -36, -11, 21, 39, 49, 39, 22, 10],  # 20N
                           [13, 12, 11, 2, -11, -28, -38, -29, -10, 3, 1, -11, -41, -42, -16, 3, 17, 33, 22, 23, 2, -3,
                            -7, -36, -59, -90, -95, -63, -24, 12, 53, 60, 58, 46, 36, 26],  # 10N
                           [22, 16, 17, 13, 1, -12, -23, -20, -14, -3, 14, 10, -15, -27, -18, 3, 12, 20, 18, 12, -13,
                            -9, -28, -49, -62, -89, -102, -63, -9, 33, 58, 73, 74, 63, 50, 32],  # 0
                           [36, 22, 11, 6, -1, -8, -10, -8, -11, -9, 1, 32, 4, -18, -13, -9, 4, 14, 12, 13, -2, -14,
                            -25, -32, -38, -60, -75, -63, -26, 0, 35, 52, 68, 76, 64, 52],  # 10S
                           [51, 27, 10, 0, -9, -11, -5, -2, -3, -1, 9, 35, 20, -5, -6, -5, 0, 13, 17, 23, 21, 8, -9,
                            -10, -11, -20, -40, -47, -45, -25, 5, 23, 45, 58, 57, 63],  # 20S
                           [46, 22, 5, -2, -8, -13, -10, -7, -4, 1, 9, 32, 16, 4, -8, 4, 12, 15, 22, 27, 34, 29, 14, 15,
                            15, 7, -9, -25, -37, -39, -23, -14, 15, 33, 34, 45],  # 30S
                           [21, 6, 1, -7, -12, -12, -12, -10, -7, -1, 8, 23, 15, -2, -6, 6, 21, 24, 18, 26, 31, 33, 39,
                            41, 30, 24, 13, -2, -20, -32, -33, -27, -14, -2, 5, 20],  # 40S
                           [-15, -18, -18, -16, -17, -15, -10, -10, -8, -2, 6, 14, 13, 3, 3, 10, 20, 27, 25, 26, 34, 39,
                            45, 45, 38, 39, 28, 13, -1, -15, -22, -22, -18, -15, -14, -10],  # 50S
                           [-45, -43, -37, -32, -30, -26, -23, -22, -16, -10, -2, 10, 20, 20, 21, 24, 22, 17, 16, 19,
                            25, 30, 35, 35, 33, 30, 27, 10, -2, -14, -23, -30, -33, -29, -35, -43],  # 60S
                           [-61, -60, -61, -55, -49, -44, -38, -31, -25, -16, -6, 1, 4, 5, 4, 2, 6, 12, 16, 16, 17, 21,
                            20, 26, 26, 22, 16, 10, -1, -16, -29, -36, -46, -55, -54, -59],  # 70S
                           [-53, -54, -55, -52, -48, -42, -38, -38, -29, -26, -26, -24, -23, -21, -19, -16, -12, -8, -4,
                            -1, 1, 4, 4, 6, 5, 4, 2, -6, -15, -24, -33, -40, -48, -50, -53, -52],  # 80S
                           [-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
                            -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30]],
                          # 90S
                          dtype=numpy.float)

wgs84_a = 6378137.0
wgs84_b = 6356752.314245
wgs84_e2 = 0.0066943799901975848
wgs84_a2 = wgs84_a ** 2  # to speed things up a bit
wgs84_b2 = wgs84_b ** 2


# 监控区域到地面的距离 11582
HEIGHT = 11582
# 简单的网格计算 1200m * 1200m
WIDTH = 1200
LENGTH = 1200
SQURE_SIZE = 75
# 光速 m/s
C = 299792458



THRESHOLD = 500

# 三个sensor的信息 也初始化为常量
# 4个传感器的序列号 分别为 依次为 322 436 13 394
# 所以这里tdoa的顺序是
sensor_322_lat = 51.833122
sensor_322_lon = 4.142814
sensor_322_height = -1.820427



sensor_436_lat = 51.341648
sensor_436_lon = 5.893123
sensor_436_height = 39.085175

sensor_13_lat = 50.862327
sensor_13_lon = 4.685708
sensor_13_height = 19.812

sensor_394_lat = 51.242813
sensor_394_lon = 6.684991
sensor_394_height = 43.901066

sensors = [322,436,13]
# sensors = [322,436,13,394]



# 75m在纬度上的距离
deleta_lon = (4.142814 - 4.141723750000001)
deleta_lat = (51.833122 - 51.83244825)


import json
# 调用库函数计算出两个坐标点的水平距离后 再计算出含有高度信息的距离
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
    distance = geodesic((lat, lon),(sensor_lat,sensor_lon )).m
    # 再计算加上高度的距离
    distance = calcDistanceContainHeight(distance,sensor_height)
    # 把km->m
    # 单位 秒
    time = distance / C
    return time



def calcTDOA(lat, lon):
    # list = []
    list = np.zeros(4,dtype=object)
    # 先计算当前网格位置离三个sensor的时间 (单位 纳秒)
    # 以A的时间为基准
    t_A = calcTime(lat,lon,sensor_322_lat,sensor_322_lon,sensor_322_height)
    t_B = calcTime(lat,lon,sensor_436_lat,sensor_436_lon,sensor_436_height)
    t_C = calcTime(lat,lon,sensor_13_lat,sensor_13_lon,sensor_13_height)
    t_D = calcTime(lat,lon,sensor_394_lat,sensor_394_lon,sensor_394_height)

    list[0] = t_A
    list[1] = t_B
    list[2] = t_C
    list[3] = t_D
    #以其中某一个为基准后 就不需要排序了
    list[1] = list[1] - list[0]
    list[2] = list[2] - list[0]
    list[0] = 0
    # 转换为array返回
    # return np.array(list)
    return list
def wgs84_height(lat, lon):
    yi = numpy.array([9 - lat / 10.0])
    xi = numpy.array([18 + lon / 10.0])
    return float(map_coordinates(wgs84_geoid, [yi, xi]))

def llh2ecef(lla):
    lat, lon, alt = lla
    lat *= (math.pi / 180.0)
    lon *= (math.pi / 180.0)

    n = lambda x: wgs84_a / math.sqrt(1 - wgs84_e2 * (math.sin(x) ** 2))

    x = (n(lat) + alt) * math.cos(lat) * math.cos(lon)
    y = (n(lat) + alt) * math.cos(lat) * math.sin(lon)
    z = (n(lat) * (1 - wgs84_e2) + alt) * math.sin(lat)

    return [x, y, z]
def llh2geoid(lla):
    # 分别从 lla中获取 纬度 经度 和高度
    lat, lon, alt = lla
    # 这里可以理解为 把 纬度 经度 和高度信息转换为了坐标信息
    (x, y, z) = llh2ecef((lat, lon, alt + wgs84_height(lat, lon)))
    return [x, y, z]

def ecef2llh(ecef):
    x, y, z = ecef
    ep = math.sqrt((wgs84_a2 - wgs84_b2) / wgs84_b2)
    p = math.sqrt(x ** 2 + y ** 2)
    th = math.atan2(wgs84_a * z, wgs84_b * p)
    lon = math.atan2(y, x)
    lat = math.atan2(z + ep ** 2 * wgs84_b * math.sin(th) ** 3, p - wgs84_e2 * wgs84_a * math.cos(th) ** 3)
    N = wgs84_a / math.sqrt(1 - wgs84_e2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - N

    lon *= (180. / math.pi)
    lat *= (180. / math.pi)

    return [lat, lon, alt]


def calcTDOAByEcef(ecef):
    '''
    根据网格的ecef坐标 和固定的传感器的ecef坐标计算 tdoa
    :param ecef: 
    :return: 
    '''
    tdoa = []

    ecef = np.array(ecef)

    ecef_322 = np.array( llh2geoid((sensor_322_lat,sensor_322_lon,sensor_322_height)))
    t_322 = np.linalg.norm( ecef - ecef_322) / C
    tdoa.append(t_322)

    ecef_436 = np.array(llh2geoid((sensor_436_lat, sensor_436_lon, sensor_436_height)))
    t_436 = np.linalg.norm(ecef - ecef_436) / C
    tdoa.append(t_436)

    ecef_13 = np.array(llh2geoid((sensor_13_lat, sensor_13_lon, sensor_13_height)))
    t_13 = np.linalg.norm(ecef - ecef_13) / C
    tdoa.append(t_13)

    ecef_394 = np.array(llh2geoid((sensor_394_lat, sensor_394_lon, sensor_394_height)))
    t_394 = np.linalg.norm(ecef - ecef_394) / C
    tdoa.append(t_394)

    tdoa[1] = tdoa[1] - tdoa[0]
    tdoa[2] = tdoa[2] - tdoa[0]
    tdoa[3] = tdoa[3] - tdoa[0]
    tdoa[0] = 0

    return np.array(tdoa)

def calcTDOAByEcef_3sensor(ecef):
    '''
    根据网格的ecef坐标 和固定的传感器的ecef坐标计算 tdoa
    :param ecef: 
    :return: 
    '''
    tdoa = []

    ecef = np.array(ecef)

    ecef_322 = np.array( llh2geoid((sensor_322_lat,sensor_322_lon,sensor_322_height)))
    t_322 = np.linalg.norm( ecef - ecef_322) / C
    tdoa.append(t_322)

    ecef_436 = np.array(llh2geoid((sensor_436_lat, sensor_436_lon, sensor_436_height)))
    t_436 = np.linalg.norm(ecef - ecef_436) / C
    tdoa.append(t_436)

    ecef_13 = np.array(llh2geoid((sensor_13_lat, sensor_13_lon, sensor_13_height)))
    t_13 = np.linalg.norm(ecef - ecef_13) / C
    tdoa.append(t_13)


    # ecef_394 = np.array(llh2geoid((sensor_394_lat, sensor_394_lon, sensor_394_height)))
    # t_394 = np.linalg.norm(ecef - ecef_394) / C
    # tdoa.append(t_394)
    tdoa[1] = tdoa[1] - tdoa[0]
    tdoa[2] = tdoa[2] - tdoa[0]
    # tdoa[3] = tdoa[3] - tdoa[0]
    tdoa[0] = 0

    return np.array(tdoa)
S2NS = 1000000000

def calcTDOAByMessage(list):
    '''
    根据message中的数据计算tdoa 时间s
    :param list: 
    :return: 
    '''

    list[1] = (list[1]/S2NS - list[0]/S2NS)
    list[2] = (list[2]/S2NS - list[0]/S2NS)
    # list[3] = (list[3]/S2NS - list[0]/S2NS)
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
    # index = distance.argsort()

    return top_k_index[0:5]

def calcLocationEstimation(top_k_index,grid):
    # 计算五个网格的 ECEF坐标的均值
    sum_X = 0
    sum_Y = 0
    sum_Z = 0
    for i in range(len(top_k_index)):
        sum_X += grid[top_k_index[i]][1]
        sum_Y += grid[top_k_index[i]][2]
        sum_Z += grid[top_k_index[i]][3]

    return [sum_X/len(top_k_index),sum_Y/len(top_k_index),sum_Z/len(top_k_index)]











def generateGridByPlaneLocation(plane_lat,plane_lon):
    '''
    根据飞机位置在1200 * 1200的范围内生成指定大小75m的网格
    传感器依然是基于 322 436 13
    :param lat: 
    :param lon: 
    :return: 
    '''

    # 首先需要根据飞机位置 计算出网格左上角的位置坐标
    deleta_lon_600 = (4.142814 - 4.134092)
    deleta_lat_600 = (51.833122 - 51.827732)

    plane_lat = plane_lat + deleta_lat_600
    plane_lon = plane_lon - deleta_lon_600


    # 固定numpy每列的属性 依次为 id X Y Z(ECEF坐标) t0 t1 t2 t3
    grid = np.zeros((0, 7), dtype=object)
    # t0 = time.time()
    # 沿着长度方向 A->C方向
    id = 0
    for i in range(int(LENGTH / SQURE_SIZE)):
        # 沿着A->B方向
        # 每一行的纬度不变 以A点为起点
        lat = plane_lat - i * deleta_lat
        for j in range(int(WIDTH / SQURE_SIZE)):
            # 每一行纬度不变 经度递增
            lon = plane_lon + j * deleta_lon
            # 计算当前网格位置的tdoa 返回值是一个list
            # 将当前网格的经纬度坐标 转换为 ECEF坐标 X Y Z
            ecef = llh2geoid((lat, lon, HEIGHT))

            tdoa = calcTDOAByEcef_3sensor(ecef)
            # 将上述信息组合为一个numpy
            temp = np.array([[id, ecef[0], ecef[1], ecef[2], tdoa[0], tdoa[1], tdoa[2]]])
            # 这里注意为了避免最后的numpy数组空一行（即第一行）,这里需要判断一下
            grid = np.vstack([grid, temp])
            id += 1
            # if id % 10000 == 0:
            #     print("已生成%.2f万条网格的tdoa" % (id / 10000))

    # print("网格信息:", grid.shape)
    # 300m 网格 单位s 3个sensor
    # np.save("Grid_75m_s_3s_ecef.npy", grid)
    # grid.to_csv("Grid_50m.csv",index=None)
    # t1 = time.time()
    # print("生成%dm网格耗时:%.2fs" % (SQURE_SIZE, t1 - t0))
    return grid





def quicklyIdentify(plane_state_lat,plane_state_lon,plane_tdoa,grid_real,threshold):
    '''
    根据飞机声称的位置 和实时网格 进行快速认证
    阈值需要变化
    :param plane_state_lat: 
    :param plane_state_lon: 
    :param grid_real: 
    :return: 
    '''
    grid_tdoa = grid_real[:,4:]
    top_k_index = calcTopKInex(plane_tdoa, grid_tdoa)

    predictEcef = calcLocationEstimation(top_k_index, grid_real)

    predictLLH = ecef2llh(predictEcef)

    error_llh = geodesic((plane_state_lat, plane_state_lon), (predictLLH[0], predictLLH[1])).m

    if error_llh > threshold:
        return 0
    else:
        return 1




def calcMeasument(TP,FP,FN,TN):

    print("准确率为:%.2f"%((TP+TN)/(TP+TN+FP+FN)))
    print("召回率为:%.2f" %((TP) / (TP + FN)))
    print("精确率为:%.2f" % ((TP) / (TP  + FP )))





if __name__ == '__main__':

    # 读入待检测的数据
    messages = pd.read_csv("DataForIdentify_Demo01.csv")

    threadholds = [500,600,700,800,900,1000]

    for temp in threadholds:
        print("开始阈值=%d的飞机合法性快速认证"%(temp))





        # 一些数据指标记录
        TP = 0 # 正确的样本 被判断为了正确
        FP = 0 # 错误的样本 被判断为了正确
        FN = 0 # 正确的样本 被判断为了错误
        TN = 0 # 错误的样本被判断为了错误

        count = 1
        # 开始遍历数据


        for index, row in messages.iterrows():
            # 获取当前消息的经纬度
            # print("当前消息")
            # print(row)
            id = row['id']
            lat = row['latitude']
            lon = row['longitude']
            height = row['geoAltitude']
            plane_llh = [lat, lon, height]
            legal = row['legal']

            # 根据飞机的位置 生成网格
            t0 = time.time()
            grid_real = generateGridByPlaneLocation(lat,lon)


            # 获取当前消息的接收数据
            # print("打印真实位置")
            # print(lat,lon)
            j = json.loads(row['measurements'])
            # 这里要注意 我们这里需要判断精确的 按照熟322,436,13,394的顺序找到三个传感器
            # 所以这里需要一个遍历
            # 同时用一个list来接收数据
            list = [0, 0, 0]
            # sensors_serial = [0,0,0]
            for i in range(len(j)):
                if sensors.__contains__(j[i][0]):
                    if j[i][0] == 322:
                        # sensors_serial[0] = j[i][0]
                        list[0] = j[i][1]
                    if j[i][0] == 436:
                        # sensors_serial[2] = j[i][0]
                        list[1] = j[i][1]
                    if j[i][0] == 13:
                        # sensors_serial[2] = j[i][0]
                        list[2] = j[i][1]

            tdoa = calcTDOAByMessage(list)

            # 开始预测
            legal_pre = quicklyIdentify(lat,lon,tdoa,grid_real,temp)
            t1 = time.time()
            # print("完成第%d条消息的快速认证，耗时=%.2f,进度为:%.2f%%"%(index+1,(t1-t0),( (index+1)/len(messages) * 100)))
            # count+=1

            if legal == 1 and legal_pre == 1:
                TP+=1
            if legal == 0 and legal_pre == 1:
                FP+=1
            if legal == 1 and legal_pre == 0:
                FN += 1
            if legal == 0 and legal_pre == 0:
                TN += 1

        # print(TP+TN)


        print("在阈值=%d的条件下,完成%d条飞机消息合法性的快速认证"%(temp,len(messages)))
        print("在阈值=%d的条件下，各项检测指标为" % temp)
        print("TP=",TP)
        print("FP=",FP)
        print("FN=", FN)
        print("TN=", TN)

        # print("预测准确率:")
        # print((TP+TN)/len(messages))


        # print("在阈值=%d的条件下，各项检测指标为"%temp)
        calcMeasument(TP,FP,FN,TN)



















