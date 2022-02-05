# -*- coding: UTF-8 -*-

'''
Author: zhuqipeng
Date: 2021-09-04 17:13:43
version: 3.5
LastEditTime: 2021-09-04 17:52:19
LastEditors: zhuqipeng
Description: 
FilePath: \Canny detect\code\main.py
'''
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.core.fromnumeric import shape
from icecream import ic
import argparse
from mpl_toolkits import mplot3d
from matplotlib.pyplot import MultipleLocator
from Gaussian_Kernel import Gaussian_Derive_Filter
from Sobel import Sobel_detect
from Prewitt import Prewitt_detect
from Canny import Canny_Detect

def set_interact_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default='./code/data/sample.jpg', type=str, required=False, help='输入图片路径')
    parser.add_argument('--result_path', default='lena.png', type=str, required=False, help='输出结果图片')
    parser.add_argument('--sigma', default=1.3, type=float, required=False, help='高斯偏导核的标准差')
    parser.add_argument('--kernel_size', default=3, type=int, required=False, help='高斯偏导核的大小')
    parser.add_argument('--high_threshold', default=0.3, type=float, required=False, help='双边阈值法的高阈值')
    parser.add_argument('--low_threshold', default=0.1, type=float, required=False, help='双边阈值法的低阈值')

    return parser.parse_args()

def Read_Img(img_address):
    return np.array(cv2.imread(img_address, 0))

def Save_Img(img, img_address):
    cv2.imwrite(img_address, img)

def main():
    '''
    description : main function
    @params : None
    return : None
    '''
    #get params
    args = set_interact_args()
    img_path = args.img_path
    result_path = args.result_path
    sigma = args.sigma
    kernel_size = args.kernel_size
    low_threshold = args.low_threshold
    high_threshold = args.high_threshold
    
    #read image
    img = Read_Img(img_path)

    #Canny Detect
    Gaussian_Derive_imgx, Gaussian_Derive_imgy, Gaussian_Derive_img, Non_maximum_img, Canny_img = Canny_Detect(img, sigma, kernel_size, low_threshold, high_threshold)

    #draw image
    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('original img')
    plt.subplot(2,2,2)
    plt.imshow(Gaussian_Derive_img, cmap='gray')
    plt.title('Gaussian_Derive img')
    plt.subplot(2,2,3)
    plt.imshow(Non_maximum_img, cmap='gray')
    plt.title('Non_maximum img')
    plt.subplot(2,2,4)
    plt.imshow(Canny_img, cmap='gray')
    plt.title('Final img')
    plt.savefig('E:/third_year_in_University/CV/experiment/Canny detect/Result/Canny_Detect' + result_path)
    plt.show()

    #Sobel Detect
    _, __, ___, ____, Sobel_img = Sobel_detect(img, sigma, kernel_size, 0.3, 0.9)

    #Prewitt Detect
    _, __, ___, ____, Prewitt_img = Prewitt_detect(img, sigma, kernel_size, 0.3, 0.9)

    #draw image
    plt.suptitle('sigma = {0}, kernelsize = {1} 各个算子对比'.format(sigma, kernel_size))
    #解决中文显示问题
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplot(1,3,1)
    plt.imshow(Canny_img, cmap='gray')
    plt.title('Canny_img')
    plt.subplot(1,3,2)
    plt.imshow(Sobel_img, cmap='gray')
    plt.title('Sobel_img')
    plt.subplot(1,3,3)
    plt.imshow(Prewitt_img, cmap='gray')
    plt.title('Prewitt_img')
    plt.show()

def test():

    # #Get Gaussian Kernel
    # Gaussian_kernel = Get_Gaussian_Kernel(2.7, 21)
    # ic(Gaussian_kernel)
    # plt.imshow(Gaussian_kernel, cmap='gray')
    # ax=plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(3))
    # ax.yaxis.set_major_locator(MultipleLocator(3))
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # X, Y = np.meshgrid(np.arange(-10,11,1), np.arange(-10,11,1)) #坐标矩阵
    # surf = ax.plot_surface(X, Y, Gaussian_kernel,  cmap=cm.Oranges)

    # # Customize the axis.
    # ax.set_zlim(0, np.max(Gaussian_kernel))
    # ax.xaxis.set_major_locator(MultipleLocator(3))
    # ax.yaxis.set_major_locator(MultipleLocator(3))

    #Get Gaussian x Derive Kernel
    # Gaussian_kernel_Derive = Get_Gaussian_Derive_x_Kernel(2.7, 21)
    # ic(Gaussian_kernel_Derive)
    # plt.imshow(Gaussian_kernel_Derive, cmap='gray')
    # ax=plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(3))
    # ax.yaxis.set_major_locator(MultipleLocator(3))
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # X, Y = np.meshgrid(np.arange(-10,11,1), np.arange(-10,11,1)) #坐标矩阵
    # surf = ax.plot_surface(X, Y, Gaussian_kernel_Derive,  cmap=cm.Oranges)

    # # Customize the axis.
    # ax.set_zlim(np.min(Gaussian_kernel_Derive), np.max(Gaussian_kernel_Derive))
    # ax.xaxis.set_major_locator(MultipleLocator(3))
    # ax.yaxis.set_major_locator(MultipleLocator(3))
    # plt.show()

    # #Get Gaussian y Derive Kernel
    # Gaussian_kernel_Derive = Get_Gaussian_Derive_y_Kernel(2.7, 21)
    # ic(Gaussian_kernel_Derive)
    # plt.imshow(Gaussian_kernel_Derive, cmap='gray')
    # ax=plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(3))
    # ax.yaxis.set_major_locator(MultipleLocator(3))
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # X, Y = np.meshgrid(np.arange(-10,11,1), np.arange(-10,11,1)) #坐标矩阵
    # surf = ax.plot_surface(X, Y, Gaussian_kernel_Derive,  cmap=cm.Oranges)

    # # Customize the axis.
    # ax.set_zlim(np.min(Gaussian_kernel_Derive), np.max(Gaussian_kernel_Derive))
    # ax.xaxis.set_major_locator(MultipleLocator(3))
    # ax.yaxis.set_major_locator(MultipleLocator(3))

    # plt.show()

    #get params
    args = set_interact_args()
    img_path = './data/1.png'
    params_list = [(1.3,3), (2.5, 7), (3, 9)]
    
    #read image
    img = Read_Img(img_path)

    #Derive Filter
    for sigma, kernel_size in params_list:
        Gaussian_Derive_imgx, Gaussian_Derive_imgy, Gaussian_Derive_img, _ = Gaussian_Derive_Filter(img, sigma, kernel_size)
        
        #画图
        plt.suptitle('sigma = {0}, kernelsize = {1}'.format(sigma, kernel_size))
        plt.subplot(2,2,1)
        plt.imshow(img, cmap='gray')
        plt.title('original img')
        plt.subplot(2,2,2)
        plt.imshow(Gaussian_Derive_imgx, cmap='gray')
        plt.title('Gaussian_Derive_imgx img')
        plt.subplot(2,2,3)
        plt.imshow(Gaussian_Derive_imgy, cmap='gray')
        plt.title('Gaussian_Derive_imgy img')
        plt.subplot(2,2,4)
        plt.imshow(Gaussian_Derive_img, cmap='gray')
        plt.title('Gaussian_Derive img')
        plt.show()

if __name__ == '__main__':
    main()
    # test()