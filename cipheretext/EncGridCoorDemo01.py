# -*- coding: utf-8 -*-
# @Time    : 2019/9/28 10:50
# @Author  : QixianZhou
# @FileName: EncGridCoorDemo01.py
# @Abstract: 加密600m网格的坐标数据 保存

import numpy as np
import time
import utils.vhe as vhe
S2NS = 1000000000
H_large = 10 ** 5
SQURE_SIZE = 600


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

if __name__ == '__main__':

    # 读取原始网格数据
    grid = np.load("Grid_600m_s_3s_ecef.npy")
    # 获取Coor(X,Y,Z)数据
    coor = grid[:, 1:4]

    # print(coor[0])


    #
    # 获取网格数据的维度 作为后续加密参数使用的一个常量
    NUMBER = coor.shape[0]
    DIMIENSION = coor.shape[1]
    # print(DIMIENSION)

    print("将原始的coor数据转换为int类型,放大系数为：%d" % (S2NS))
    coor_int = np.zeros((NUMBER, DIMIENSION), dtype=object)
    for i in range(NUMBER):
        coor_int[i] = float2Int(coor[i])
    print("检查数据类型是否成功转换")
    if type(coor_int[0][0]) == int:
        print("成功把浮点型的tdoa转换为python原生的int类型")

    # print(coor_int[0])

    #
    print("初始化加密参数")
    # 初始化相关加密参数 注意这里的加密方式不是原生的VHE的加密，而是为了求欧式距离，特殊定义了一些加密方式
    T = vhe.getRandomMatrix(DIMIENSION, 1, vhe.tBound)
    S = vhe.getSecretKey(T)
    I = np.eye(DIMIENSION, dtype=object)
    M = vhe.KeySwitchMatrix(I, T)

    # M可以用来进行加密了
    # 先定义一个numpy数组 来存储加密数据
    enc_coor = np.zeros((NUMBER, DIMIENSION + 1), dtype=object)
    print("开始对%dm网格的%d条coor数据进行加密" % (SQURE_SIZE, NUMBER))
    t0 = time.time()
    for i in range(NUMBER):
        enc_coor[i] = vhe.encrypt_distance(M, coor_int[i])

        if (i + 1) % 10000 == 0:
            print("已完成%d条数据的加密，当前进度为%.2f%%" % ((i + 1), ((i + 1) / NUMBER) * 100))
    #
    t1 = time.time()
    print("完成加密，耗时%.2f秒" % (t1 - t0))
    print("保存密文coor")
    np.save("EncCOOR_600m.npy", enc_coor)

    print("保存密钥S,M")
    np.save("EncCOOR_600m_S.npy", S)
    np.save("EncCOOR_600m_M.npy", M)



