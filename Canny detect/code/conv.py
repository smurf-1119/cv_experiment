'''
Author: zhuqipeng
Date: 2021-09-04 16:58:59
version: 3.5
LastEditTime: 2021-09-04 16:58:59
LastEditors: zhuqipeng
Description: 
FilePath: \Canny detect\conv.py
'''
import numpy as np

def Filp_Img(img, flag):
    '''
    description: Flip the image, 
                if flag = 1, flip the image in vertical direction;
                if flag = 0, flip the image in horizontal direction;
                if flag = -1, flip the image in vertical and horizontal directions.
    @params: {array[[]]} img
    @params: {int} flag
    return: {array[[]]} result_img
    '''
    (row, col) = img.shape
    center = (row//2, col//2)
    new_img = np.copy(img)

    if(flag == 1):
        for i in range(row):
            for j in range(col//2):
                new_img[i, j], new_img[i, 2 * center[1] - j] = new_img[i, 2 * center[1] - j], new_img[i, j]
    elif(flag == 0):
        for i in range(row//2 + 1):
            for j in range(col):
                new_img[i, j], new_img[2 * center[0] - i, j] = new_img[2 * center[0] - i, j], new_img[i, j]
    else:
        for i in range(row//2):
            for j in range(col):
                new_img[i, j], new_img[2 * center[0] - i, 2 * center[1] - j] = new_img[2 * center[0] - i, 2 * center[1] - j], new_img[i, j]
        for j in range(col // 2):
            new_img[center[0], j], new_img[center[0], 2 * center[1] - j] = new_img[center[0], 2 * center[1] - j], new_img[center[0], j]
    return new_img

def conv(img, kernel):
    '''
    description: image convolution function, input a two-dim image and a two-dim kernel
    @params: {array[[]]} img
    @params: {array[[]]} kernel
    return: {array[[]]} result image after convolution
    '''

    (img_row, img_col) = img.shape #get the original image shape
    (kernel_row, kernel_col) = kernel.shape # get the kernel shape
    kernel_flip = Filp_Img(kernel, -1) #Flip the original image
    strading_img = np.zeros((img_row + kernel_row - 1, img_col + kernel_col - 1)) 
    strading_img[(kernel_row - 1)//2:img_row + (kernel_row - 1)//2, 
    (kernel_col - 1)//2:img_col + (kernel_col - 1)//2] = img[:, :] #initialize the strading image
    result_img = np.zeros((img_row, img_col)) #initialize the result image with zeros
    
    #Convolution
    for i in range(img_row):
        for j in range(img_col):
            result_img[i, j] = np.sum(kernel_flip * strading_img[i:i + kernel_row, j:j + kernel_col])
            
    return result_img