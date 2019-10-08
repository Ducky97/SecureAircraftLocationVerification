# -*- coding: utf-8 -*-
# @Time    : 2019/9/16 20:10
# @Author  : QixianZhou
# @FileName: LocationEstimationDemo06.py
# @Abstract: 位置预测算法 Demo2 少量数据 采用numpy存储的网格  结果也采用numpy存储
# 20190917 基于600m网格时间精度为 秒 预测数据
# 修改tdoa算法后的测试类
# 基于ecef坐标系的预测算法
# 采用4个sensor 322,436,13,394

import numpy
import numpy as np
import pandas as pd
import json
import pickle
import math
from scipy.ndimage import map_coordinates
# 飞机的tdoa需要按照这个顺序生成
sensors = [322,436,13,394]
# 秒到纳秒的换算
S2NS = 1000000000
HEIGHT = 11 * 1000
C = 299792458
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
    根据message中的数据计算tdoa 时间s
    :param list: 
    :return: 
    '''

    list[1] = (list[1]/S2NS - list[0]/S2NS)
    list[2] = (list[2]/S2NS - list[0]/S2NS)
    list[3] = (list[3]/S2NS - list[0]/S2NS)
    list[0] = 0
    return np.array(list)
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

def calcTop_1_Inex(tdoa,grid_numpy):
    # print("---------")
    # print(grid_numpy.shape[0])


    # 先计算tdoa和grid_numpy的每一条记录的欧式距离
    distance = np.zeros(grid_numpy.shape[0])
    for i in range(distance.shape[0]):
        distance[i] = np.linalg.norm(tdoa - grid_numpy[i])
    # 计算topk 并返回索引
    # top_k_index = np.argpartition(distance, 1)

    return index[0:5]



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
    grid = np.load("Grid_600m_s_4s_ecef.npy")


    grid_tdoa = grid[:,4:]
    print(grid_tdoa.shape)


    # 读入飞机位置数据
    messages = pd.read_csv("MessagesReceivedByRadarcapeBy3_11w_322_394_436_13_MonitorArea0.csv")
    # print(messages.info())
    print("消息条数:",len(messages))

    # 新建dataframe 记录消息的预测情况
    # id是原消息的id 真实位置 预测位置 误差
    predictError = pd.DataFrame(columns=['id','latitude','longitude','height','predictLatitude','predictLongitude','predictHeight','error'])

    count = 0
    # 记录每一次的 error_llh 用于最后计算RMSE
    error = []
    for index,row in messages.iterrows():
        # 获取当前消息的经纬度
        # print("当前消息")
        # print(row)
        id = row['id']
        lat = row['latitude']
        lon = row['longitude']
        height = row['geoAltitude']
        plane_llh = [lat,lon,height]



        # 获取当前消息的接收数据
        # print("打印真实位置")
        # print(lat,lon)
        j = json.loads(row['measurements'])
        # 这里要注意 我们这里需要判断精确的 按照熟322,436,13,394的顺序找到三个传感器
        # 所以这里需要一个遍历
        # 同时用一个list来接收数据
        list = [0,0,0,0]
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
                if j[i][0] == 394:
                    list[3] = j[i][1]



        # print("查看飞机的tdoa前的数据")
        # print(sensors_serial)
        # print(list)
        # print("查看挑选的list，是不是对应的sensor的数据")
        # print(list)
        # 计算tdoa
        tdoa = calcTDOAByMessage(list)
        # 拿到tdoa后 可以开始预测位置了
        # 这里返回的是topk的索引 然后根据索引再去grid表中找对应的位置信息 再求平均值
        top_k_index = calcTopKInex(tdoa,grid_tdoa)
        # top_k_index = calcTop_1_Inex(tdoa,grid_numpy)
        # print("打印topk索引")
        # print(top_k_index)
        #打印输出最近的五个点
        # print("******************************************")
        # print("当前飞机的位置和tdoa为：",lat,lon,tdoa)
        # print("******************************************")
        # print("打印输出预测出的top5个网格,以及这5个网格的相关信息")
        # for x in top_k_index:
        #     print("当前网格的序号为：",x)
        #     print("当前网格的ECEF位置和tdoa为",grid[x][1:4],grid[x][4:])
        #     tdoa_eula = np.linalg.norm(grid_tdoa[x] - tdoa)
        #     print("当前网格的tdoa和飞机的tdoa的欧式距离为：",tdoa_eula)
        #     # error = geodesic((lat, lon), (grid[x][1], grid[x][2])).m
        #     # print("当前网格与飞机的真实位置距离为%.2fm" % ( error))
        #     print("************************************************")






        # 根据索引，计算预测位置的ECEF值
        predictEcef = calcLocationEstimation(top_k_index,grid)
        print("打印预测ECEF位置")
        print(predictEcef)
        plane_ecef = llh2geoid(plane_llh)
        print("打印飞机的ECEF位置")
        print(plane_ecef)

        error_ecef = np.linalg.norm( np.array(predictEcef) - np.array(plane_ecef) )
        print("二者在ECEF坐标下的误差为%.2fm"%error_ecef)
        # print(error_ecef,"m")

        print("预测的ECEF位置转换为经纬度和高度")
        predictLLH = ecef2llh(predictEcef)
        print(predictLLH)

        error_llh = geodesic((lat, lon), (predictLLH[0], predictLLH[1])).m
        print("预测的LLH位置和飞机的LLH位置的误差为%.2fm"%error_llh)
        error.append(error_llh)
        # 计算误差
        # error = geodesic((lat, lon), ( predictLatitude ,  predictLongitude )).m
        # print("打印预测误差")
        # print(error)
        # # 保存数据
        predictError =  predictError.append([{'id': id, 'latitude': lat, "longitude": lon,'height':height, 'predictLatitude': predictLLH[0],'predictLongitude': predictLLH[1],'predictHeight':predictLLH[2],'error': error_llh}])
        #
        count+=1
        # # print("完成%d条消息预测"%count)
        # # # 测试100条数据
        # print("---------------------------------------------------------")
        if count %100 == 0:
            print("---------------------------------完成%d条消息预测----------------------------------------" % count)
        # break


    print("完成%d条消息的预测,预测基于4个传感器生成的600m网格" % count)
    mean = predictError['error'].mean()
    median = predictError['error'].median()
    rmse = calcRMSE(error)
    print("预测误差的均值为:", mean, "m,中位值为:", median, "m,RMSE为:", rmse, "m")

    numOfGreater1000 = 0

    for index, row in predictError.iterrows():
        # error.append(row['error'])
        if row['error'] >= 1000:
            numOfGreater1000 += 1
    print("在：%d条预测误差中,预测误差中大于1000m的消息数量为：%d,小于1000m的消息数量为：%d"%(len(messages),numOfGreater1000,len(messages) - numOfGreater1000))
    rate_1 = numOfGreater1000/len(messages)
    rate_2 = 1 - rate_1
    print("其中预测误差大于1000m的消息数量占比为%.2f,小于1000m的消息数量占比为%.2f"%(rate_1,rate_2))
    print("开始保存预测数据")
    predictError.to_csv("predictError_Grid_600m_s_4sensor_ecef.csv",index=None)


