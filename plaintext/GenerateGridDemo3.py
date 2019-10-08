# -*- coding: utf-8 -*-
# @Time    : 2019/9/11 20:30
# @Author  : QixianZhou
# @FileName: GenerateGridDemo1.py
# @Abstract: 以500 * 500的区域做一个简单的demo


import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from geopy.distance import geodesic
from geopy.distance import lonlat
import geopy
import geopy.distance
import math
from geographiclib.geodesic import Geodesic

# 监控区域到地面的距离 11km
HEIGHT = 11 * 1000
# 简单的网格计算 500m * 500m
WIDTH = 250
LENGTH = 500
SQURE_SIZE = 50
# 光速 m/s
C = 299792458

# 三个sensor的信息 也初始化为常量
sensor_A_lat = 51.833122
sensor_A_lon = 4.142814
sensor_A_height = -1.820427

sensor_B_lat = 51.242813
sensor_B_lon = 6.684991
sensor_B_height = 43.901066

sensor_C_lat = 51.341648
sensor_C_lon = 5.893123
sensor_C_height = 39.085175

import json
# 调用库函数计算出两个坐标点的水平距离后 再计算出含有高度信息的距离
def calcDistanceContainHeight(point_distance,sensor_height):
    '''
    
    :param point_distance: 两点之间的距离
    :param sensor_height: sensor的高度
    :return: 
    '''
    height = HEIGHT - sensor_height
    sum = point_distance**2 + height**2
    return math.sqrt(sum)


def calcTime(lat,lon,sensor_lat,sensor_lon,sensor_height):
    # 先计算两点的距离
    distance = geodesic((lat, lon), ( sensor_lat ,  sensor_lon )).m
    # 再计算加上高度的距离
    distance = calcDistanceContainHeight(distance,sensor_height)
    # 把km->m
    # 单位 秒
    time = distance / C
    # 换成 纳秒
    return time * 1000000000



def calcTDOA(lat, lon):
    list = []
    # 先计算当前网格位置离三个sensor的时间 (单位 纳秒)
    t_A = calcTime(lat,lon,sensor_A_lat,sensor_A_lon,sensor_A_height)
    t_B = calcTime(lat,lon,sensor_B_lat,sensor_B_lon,sensor_B_height)
    t_C = calcTime(lat,lon,sensor_C_lat,sensor_C_lon,sensor_C_height)
    list.append(t_A)
    list.append(t_B)
    list.append(t_C)
    # 排序
    list.sort()
    list[1] = list[1] - list[0]
    list[2] = list[2] - list[0]
    list[0] = 0
    # 转换为array返回
    # return np.array(list)
    return list










# def get_distance_point(lat, lon, distance, direction):
#     """
#     根据经纬度，距离，方向获得一个地点
#     :param lat: 纬度
#     :param lon: 经度
#     :param distance: 距离（千米）
#     :param direction: 方向（北：0，东：90，南：180，西：360）
#     :return:
#     """
#     start = geopy.Point(lat, lon)
#     d = geopy.distance.VincentyDistance(kilometers=distance)
#
#     return d.destination(point=start, bearing=direction)



# 50m在经度上的距离
deleta_lon_50 = (4.142814-4.142087)
# 50m在纬度上的距离
deleta_lat_50 = (51.833122-51.833571)

if __name__ == '__main__':
    # lat纬度 lon 经度
    lat_A =  51.833122
    lon_A = 4.142814

    lat_B = 51.833122
    lon_B = 6.142814

    lat_C = 49.833122
    lon_C = 4.142814

    lat_D = 49.833122
    lon_D = 6.142814

    grid = pd.DataFrame(columns=['id', 'latitude', 'longitude', 't0','t1','t2'])
    # grid = grid.append([{'id': count, 'latitude': lat_A, "longitude": lon_B, 'tdoa': data}])


    # 沿着长度方向 A->C方向
    id = 0
    for i in range(int(LENGTH/SQURE_SIZE)):
        #沿着A->B方向
        # 每一行的纬度不变 以A点为起点
        lat = lat_A - i * deleta_lat_50
        for j in range(int(WIDTH/SQURE_SIZE)):
            # 每一行纬度不变 经度递增
             lon = lon_A + j *deleta_lon_50
             # 计算当前网格位置的tdoa 返回值是一个array
             tdoa = calcTDOA(lat,lon)
             # 将id 信息和 位置信息 添加到grid中
             grid = grid.append([{'id': id, 'latitude': lat, "longitude": lon, 't0': tdoa[0],'t1':tdoa[1],'t2':tdoa[2]}])
             id+=1


    # 保存
    grid.to_csv("Grid_50m_test.csv",index=None)








