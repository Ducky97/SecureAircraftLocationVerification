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
from geographiclib.geodesic import Geodesic

def get_distance_point(lat, lon, distance, direction):
    """
    根据经纬度，距离，方向获得一个地点
    :param lat: 纬度
    :param lon: 经度
    :param distance: 距离（千米）
    :param direction: 方向（北：0，东：90，南：180，西：360）
    :return:
    """
    start = geopy.Point(lat, lon)
    d = geopy.distance.VincentyDistance(kilometers=distance)
    return d.destination(point=start, bearing=direction)



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

    # 计算 A-> B的方向角
    # print(Geodesic.WGS84.Inverse(lat_A, lon_A, lat_B, lon_B))
    # direction_A_B = 89.21375517995789
    # 90.78624482004211
    direction_A_B = 90.78624482004211
    # 根据A的坐标 往A-B方向前进50m 后的坐标
    a = get_distance_point(lat_A,lon_A,0.05,direction_A_B)
    print(a.latitude)
    print(a.longitude)



    #
    # Lat_B =  51.833122
    # Lng_B = Lng_A + deleta_lon_50
    # height_B = 11


    # # 参数顺序是 纬度(Latitude)经度(Longtitude)
    distance = geodesic((lat_A, lon_A), ( a.latitude ,  a.longitude )).km
    #
    print(distance)


