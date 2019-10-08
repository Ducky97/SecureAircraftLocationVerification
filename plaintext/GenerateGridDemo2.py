# -*- coding: utf-8 -*-
# @Time    : 2019/9/11 20:30
# @Author  : QixianZhou
# @FileName: GenerateGridDemo1.py
# @Abstract: 


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
HEIGHT = 11
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

    data = np.arange(1,10).reshape(3,3)
    # data = np.zeros((2,3))
    # data
    print(data)

    # index = np.


    # # 创建dataframe 用来存储 id latitude longtitude tdoa
    # grid = pd.DataFrame(columns=['id','latitude','longitude','tdoa'])
    # data = np.array([0, 1, 2])
    # count = 1
    # grid = grid.append([{'id': count, 'latitude': lat_A, "longitude": lon_B, 'tdoa': data}])
    # grid.to_csv("Grid_50m.csv")




    # # 读取csv文件 这里是因为直接创建比较麻烦
    # grid = pd.read_csv("Grid_50.csv")
    # # print(grid.head())
    # # print(grid.info())
    #
    # print(type(grid['tdoa'][0]))
    # tdoa = grid['tdoa'][0]
    # temp = []
    # for x in tdoa:
    #     n1 = eval(x)
    #     temp.append(n1)
    # data = np.array(eval(tdoa))
    # print(data)
    #
    # print(data.shape)
    #
    #
    #
    # print(data.tostring())
    #
    #
    # data = np.array([0,1,2])
    #
    # # 可以直接把 np.array对象 存进去
    # grid = grid.append([{'id':2,'latitude':111,"longitude":222,'tdoa':data}])
    #
    # grid.to_csv("Grid_50m.csv")






    # grid = pd.DataFrame(columns=['id','latitude','longitude','tdoa'])
    #
    # new = pd.DataFrame({"1","2","3","[0,1,2]"},index=["0"])
    #
    # grid.append(new)


    # a = np.array([1,2,3])



    # grid.to_csv("Grid_50m.csv",index=None)








