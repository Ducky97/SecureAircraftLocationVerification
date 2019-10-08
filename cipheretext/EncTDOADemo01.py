# -*- coding: utf-8 -*-
# @Time    : 2019/9/28 10:57
# @Author  : QixianZhou
# @FileName: EncTDOADemo01.py
# @Abstract:

import numpy as np
import utils.vhe as vhe
import time


# 尝试把tdoa放大这么多倍
S2NS = 1000000000
H_large = 10 ** 5
SQURE_SIZE = 600


def float2Int(float_tdoa):
    '''
    把原始的数据类型 转换为 python原生的数据类型
    
    :param data: 
    :return: 
    '''
    int_tdoa = np.zeros(len(float_tdoa),dtype=object)

    for i in range(len(float_tdoa)):
        int_tdoa[i] = int(float_tdoa[i] * S2NS)

    return int_tdoa







if __name__ == '__main__':

    # 读取原始网格数据
    grid = np.load("Grid_600m_s_3s_ecef.npy")
    # 获取tdoa数据
    tdoa = grid[:,4:]

    # 获取网格数据的维度 作为后续加密参数使用的一个常量
    NUMBER = tdoa.shape[0]
    DIMIENSION = tdoa.shape[1]
    # print(DIMIENSION)

    print("将原始的tdoa数据转换为int类型,放大系数为：%d"%(S2NS))
    tdoa_int = np.zeros((NUMBER,DIMIENSION),dtype=object)
    for i in range(NUMBER):
        tdoa_int[i] = float2Int(tdoa[i])
    print("检查数据类型是否成功转换")
    if type(tdoa_int[0][0]) == int:
        print("成功把浮点型的tdoa转换为python原生的int类型")


    print("初始化加密参数")
    # 初始化相关加密参数 注意这里的加密方式不是原生的VHE的加密，而是为了求欧式距离，特殊定义了一些加密方式
    T = vhe.getRandomMatrix(DIMIENSION,1,vhe.tBound)
    S = vhe.getSecretKey(T)
    I = np.eye(DIMIENSION,dtype=object)
    M = vhe.KeySwitchMatrix(I,T)

    # M可以用来进行加密了
    # 先定义一个numpy数组 来存储加密数据
    enc_tdoa = np.zeros((NUMBER,DIMIENSION + 1),dtype=object)
    print("开始对%dm网格的%d条tdoa数据进行加密"%(SQURE_SIZE,NUMBER))
    t0 = time.time()
    for i in range(NUMBER):
        enc_tdoa[i] = vhe.encrypt_distance(M,tdoa_int[i])

        if (i+1)%10000 == 0:
            print("已完成%d条数据的加密，当前进度为%.2f%%"%((i+1),((i+1)/NUMBER) * 100))

    t1 = time.time()
    print("完成加密，耗时%.2f秒"%(t1 - t0))
    print("保存密文tdoa")
    np.save("EncTDOA_600m.npy",enc_tdoa)


    print("开始基于加密参数生成H矩阵")
    M_float = np.array(M,dtype=float)
    M_I = np.linalg.pinv(M_float)
    I_star = vhe.getBitMatrix(I)
    A = I_star.dot(M_I)
    H = A.T.dot(A)
    H_int = np.zeros((H.shape[0],H.shape[1]),dtype=object)

    print("H矩阵整数化，放大系数为：%d"%(H_large))
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            H_int[i][j] = int(H_large * H[i][j])
    print("检查数据类型是否成功转换")
    if type(H_int[0][0]) == int:
        print("成功把浮点型的H矩阵转换为python原生的int类型")

    print("保存S,M,H矩阵")
    np.save("EncTDOA_600m_S.npy", S)
    np.save("EncTDOA_600m_M.npy",M)
    np.save("EncTDOA_600m_H.npy", H_int)














