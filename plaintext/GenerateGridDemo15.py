# -*- coding: utf-8 -*-
# @Time    : 2019/9/11 20:30
# @Author  : QixianZhou
# @FileName: GenerateGridDemo14.py
# @Abstract: 网格的存储尝试用numpy 用全部数据
#   固定numpy每列的属性 依次为 id 纬度 经度 t0 t1 t2
#   避免漫长的等待 先跑个600m的网格试试  时间差 秒 精度
# 修改tdoa的算法 固定一个传感器322 即下文变量中的A 为基准
# 进一步修改 不用tdoa  就用达到时间 作为指纹向量


import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from geopy.distance import geodesic
from geopy.distance import lonlat
import geopy
import geopy.distance
import math
import time


from geographiclib.geodesic import Geodesic

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
    list = np.zeros(3,dtype=object)
    # 先计算当前网格位置离三个sensor的时间 (单位 纳秒)
    # 以A的时间为基准
    t_A = calcTime(lat,lon,sensor_A_lat,sensor_A_lon,sensor_A_height)
    t_B = calcTime(lat,lon,sensor_B_lat,sensor_B_lon,sensor_B_height)
    t_C = calcTime(lat,lon,sensor_C_lat,sensor_C_lon,sensor_C_height)

    list[0] = t_A
    list[1] = t_B
    list[2] = t_C
    # 不用tdoa就用达到时间
    #以其中某一个为基准后 就不需要排序了
    # list[1] = list[1] - list[0]
    # list[2] = list[2] - list[0]
    # list[0] = 0
    # 转换为array返回
    return list
    # return list






# 600m在纬度上的距离
deleta_lon = (4.142814-4.134092)
deleta_lat = (51.833122-51.827732)

if __name__ == '__main__':
    # lat纬度 lon 经度
    # 网格起始位置
    lat_A =  51.833122
    lon_A = 4.142814
    #存储每一个网格的信息
    #固定numpy每列的属性 依次为 id 纬度 经度 t0 t1 t2
    grid = np.zeros((0,6),dtype=object)
    t0= time.time()
    # 沿着长度方向 A->C方向
    id = 0
    for i in range(int(LENGTH/SQURE_SIZE)):
        #沿着A->B方向
        # 每一行的纬度不变 以A点为起点
        lat = lat_A - i * deleta_lat
        for j in range(int(WIDTH/SQURE_SIZE)):
            # 每一行纬度不变 经度递增
             lon = lon_A + j * deleta_lon
             # 计算当前网格位置的tdoa 返回值是一个list
             tdoa = calcTDOA(lat,lon)
             # 将上述信息组合为一个numpy
             temp = np.array([[id,lat,lon,tdoa[0],tdoa[1],tdoa[2]]])
             # 这里注意为了避免最后的numpy数组空一行（即第一行）,这里需要判断一下
             grid = np.vstack([grid,temp])
             id+=1
             if id%10000 == 0:
                 print("已生成%.2f万条网格的tdoa"%(id/10000))


    print("网格信息:" ,grid.shape)
    # print(grid)
    # 保存
    # ns代表tdoa精度为 s秒
    np.save("Grid_600m_s_NonTDOA.npy",grid)
    # grid.to_csv("Grid_50m.csv",index=None)
    t1 = time.time()
    print("生成%dm网格耗时:%.2fs"%(SQURE_SIZE,t1-t0))







