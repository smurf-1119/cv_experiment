'''
Author: zhuqipeng
Date: 2021-09-04 17:10:07
version: 3.5
LastEditTime: 2021-09-04 17:10:08
LastEditors: zhuqipeng
Description: 
FilePath: \Canny detect\Hysteresis_thresholding.py
'''
import numpy as np

def Hysteresis_thresholding(img, low_rate, high_rate):
    '''
    description: 
    @param {array[[]]} img
    @param {float} low_rate
    @param {float} high_rate
    return {array[[]]} flag_mat[1:img.shape[0]+1, 1:img.shape[1]+1]
    '''
    max_pixel = np.max(img)
    low, high = low_rate*max_pixel, high_rate*max_pixel
    flag_mat = np.zeros((img.shape[0]+2, img.shape[1]+2))
    new_img = np.zeros((img.shape[0]+2, img.shape[1]+2))
    new_img[1:img.shape[0]+1, 1:img.shape[1]+1] = np.copy(img)
    q1 = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if new_img[i+1,j+1] < low:
                flag_mat[i+1,j+1] = 0
            elif new_img[i+1,j+1] > high:
                flag_mat[i+1,j+1] = 255
            else:
                q1.append((i+1,j+1))
                flag_mat[i+1,j+1] = -1

    
    while(True):
        q2 = []
        flag = 0 #用于记录是否有对矩阵更改
        while(len(q1) != 0):
            (i,j) = q1.pop()
            if(flag_mat[i-1,j-1] or flag_mat[i-1,j] or flag_mat[i-1,j+1] or flag_mat[i,j-1] or flag_mat[i,j+1] or flag_mat[i+1,j-1] or flag_mat[i+1,j] or flag_mat[i+1,j+1]):
                flag_mat[i,j] = 255
                flag = 1
            elif(flag_mat[i-1,j-1] == 0 and flag_mat[i-1,j] == 0 and flag_mat[i-1,j+1] == 0 and flag_mat[i,j-1] == 0 and flag_mat[i,j+1] == 0 and flag_mat[i+1,j-1] == 0 and flag_mat[i+1,j] == 0 and flag_mat[i+1,j+1] == 0):
                flag_mat[i,j] = 0
            else:
                q2.append((i,j))
        
        if(len(q2) != 0 or flag == 1):
            q1 = q2[:]
        else:
            break
    
    flag_mat = np.array(flag_mat)
    flag_mat[flag_mat==-1] = 0

    return flag_mat[1:img.shape[0]+1, 1:img.shape[1]+1]
