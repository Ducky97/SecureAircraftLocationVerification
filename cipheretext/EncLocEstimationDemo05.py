# -*- coding: utf-8 -*-
# @Time    : 2019/9/28 14:14
# @Author  : QixianZhou
# @FileName: EncLocEstimationDemo03.py
# @Abstract: 600m网格的密文预测

import numpy
import numpy as np
import utils.vhe as vhe
import math
import time
import pandas as pd
import json
from scipy.ndimage import map_coordinates
from geopy.distance import geodesic
# 一些常量信息
SQURE_SIZE = 50
SENSORS = [322,436,13]
S2NS = 1000000000
DIMENSION = 3

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

def calcRMSE(error):
    '''
    定义计算均方根误差的函数
    :param error: 
    :return: 
    '''

    # if type(error) != list:
    #     raise "the paramater type is not list!please check!"
    mse = sum([x ** 2 for x in error]) / len(error)
    return  math.sqrt(mse)


def calcTDOAByMessage(list):
    '''
    根据message中的数据计算tdoa 时间 
    :param list: 
    :return: 
    '''

    list[1] = (list[1]/S2NS - list[0]/S2NS)
    list[2] = (list[2]/S2NS - list[0]/S2NS)
    list[0] = 0
    return np.array(list)


def float2Int(float_data):
    '''
    把原始的数据类型 转换为 python原生的数据类型

    :param data: 
    :return: 
    '''
    int_data = np.zeros(len(float_data), dtype=object)

    for i in range(len(float_data)):
        int_data[i] = int(float_data[i] * S2NS)

    return int_data

def calcEncEuclideanDistance(vec1,vec2,H):
    '''
    计算vec1 和 vec2在密文下的欧式距离
    :param vec1: 
    :param vec2: 
    :param H: 
    :return: 
    '''
    distance = ((vec2 - vec1).T.dot(H)).dot(vec2 - vec1)
    return distance






def calcEncTopKInex(enc_plane_tdoa_int, EncTDOA,EncTDOA_H):

    distance = np.zeros(EncTDOA.shape[0])
    for i in range(EncTDOA.shape[0]):
        distance[i] = calcEncEuclideanDistance(enc_plane_tdoa_int,EncTDOA[i],EncTDOA_H)

    top_k_index = np.argpartition(distance, 5)
    return top_k_index[0:5]

def calcEncLocationEstimationSum(top_k_index,encCoor):
    # 计算五个网格的 ECEF坐标的均值

    encSumCoor = np.zeros((1,encCoor.shape[1]),dtype=object)
    for i in range(len(top_k_index)):
       encSumCoor += encCoor[top_k_index[i]]

    return encSumCoor.T

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

if __name__ == '__main__':



    print("系统初始化，读取飞机数据以及相关密文网格数据")
    # 读取飞机消息数据
    messages = pd.read_csv("Messages_50m_500.csv")

    # 读取网格的密文tdoa 和相关数据
    EncTDOA = np.load("EncTDOA_50m.npy")
    EncTDOA_M = np.load("EncTDOA_50m_M.npy")
    EncTDOA_H = np.load("EncTDOA_50m_H.npy")
    EncTDOA_S = np.load("EncTDOA_50m_S.npy")

    # 读取网格的密文位置和密钥
    EncCOOR = np.load("EncCOOR_50m.npy")
    EncCOOR_S = np.load("EncCOOR_50m_S.npy")

    predictError = pd.DataFrame(columns=['id', 'latitude', 'longitude', 'height', 'predictLatitude', 'predictLongitude', 'predictHeight','error'])

    print("开始基于%dm的网格数据对%d条飞机消息进行密文下位置预测"%(SQURE_SIZE,len(messages)))
    t0 = time.time()
    # 记录误差数据，方便后文统计计算
    error = []
    for index,row in messages.iterrows():
        id = row['id']
        lat = row['latitude']
        lon = row['longitude']
        height = row['geoAltitude']

        # 按sensors顺序获取飞机的tdoa
        j = json.loads(row['measurements'])
        plane_toa = [0, 0, 0]
        # sensors_serial = [0,0,0]
        for i in range(len(j)):
            if SENSORS.__contains__(j[i][0]):
                if j[i][0] == 322:
                    plane_toa[0] = j[i][1]
                if j[i][0] == 436:
                    plane_toa[1] = j[i][1]
                if j[i][0] == 13:
                    plane_toa[2] = j[i][1]

        plane_tdoa = calcTDOAByMessage(plane_toa)

        # 将浮点型的飞机tdoa转换为整型
        plane_tdoa_int = float2Int(plane_tdoa)
        enc_plane_tdoa_int = vhe.encrypt_distance(EncTDOA_M,plane_tdoa_int)

        # 加密的飞机的tdoa和网格的tdoa逐条计算密文欧式距离求top5的index
        top_k_index = calcEncTopKInex(enc_plane_tdoa_int, EncTDOA,EncTDOA_H)

        # 根据索引 再Coor中寻找对应的坐标数据累加
        encSumCoor = calcEncLocationEstimationSum(top_k_index,EncCOOR)
        # 对密文位置和解密
        decSumCoor = vhe.decrypt(EncCOOR_S,encSumCoor)
        # 解密后再缩小放大倍数  再取均值
        predictCoor = decSumCoor / (5*S2NS)
        # 然后转换为llh坐标
        predictLLH = ecef2llh(predictCoor)

        error_llh = geodesic((lat, lon), (predictLLH[0], predictLLH[1])).m
        print("预测的LLH位置和飞机的LLH位置的误差为%.2fm" % error_llh)
        error.append(error_llh)
        predictError = predictError.append([{'id': id, 'latitude': lat, "longitude": lon, 'height': height,'predictLatitude': predictLLH[0], 'predictLongitude': predictLLH[1],'predictHeight': predictLLH[2], 'error': error_llh}])
        if (index + 1)%10 == 0:
            print("完成%d条消息预测，进度为:%.2f%%" %(index+1,((index+1)/len(messages)) * 100))


    t1 = time.time()
    print("完成密文下%d条消息的预测,预测基于%d个传感器生成的%dm密文网格" % (len(messages), len(SENSORS), SQURE_SIZE))
    print("耗时：%.2f"%(t1 - t0))
    mean = predictError['error'].mean()
    median = predictError['error'].median()
    rmse = calcRMSE(error)
    print("预测误差的均值为:", mean, "m,中位值为:", median, "m,RMSE为:", rmse, "m")

    print("开始保存预测数据")
    predictError.to_csv("EncPredictError_Grid_50m_s_3sensor_ecef.csv", index=None)





