# coding=utf-8
'''
Author: zhuqipeng
Date: 2021-08-22 22:38:51
version: 3.5
LastEditTime: 2021-09-04 17:16:00
LastEditors: zhuqipeng
Description: Edge Detect Algorithm
params: 
    '--img_path', default='./data/sample.jpg', type=str, required=False, help='输入图片路径'
    '--result_path', default='lena.png', type=str, required=False, help='输出结果图片'
    '--sigma', default=1.3, type=float, required=False, help='高斯偏导核的标准差'
    '--kernel_size', default=3, type=int, required=False, help='高斯偏导核的大小'
    '--high_threshold', default=0.3, type=float, required=False, help='双边阈值法的高阈值'
    '--low_threshold', default=0.1, type=float, required=False, help='双边阈值法的低阈值'
FilePath: \Canny detect\Canny_detect.py
'''
from Gaussian_Kernel import Gaussian_Derive_Filter
from Hysteresis_thresholding import Hysteresis_thresholding
from Non_maximum import Non_maximum

def Canny_Detect(img, sigma, kernelsize, min_threshold = 0.1, max_threshold = 0.3):
    '''
    description: Extract the edge of the picture
    @param {array[[]]} img
    @param {float} sigma
    @param {int} kernelsize
    @param {float} min_threshold
    @param {float} max_threshold
    return {array[[]]} Gaussian_Derive_imgx 
    return {array[[]]} Gaussian_Derive_imgy
    return {array[[]]} Gaussian_Derive_img
    return {array[[]]} Non_maximum_img
    return {array[[]]} final_img
    '''
    #use Gaussian Derive Filter to Filter the image first
    Gaussian_Derive_imgx, Gaussian_Derive_imgy, Gaussian_Derive_img, dir = Gaussian_Derive_Filter(img, sigma, kernelsize)

    #refine the image
    Non_maximum_img = Non_maximum(Gaussian_Derive_img, dir)

    #use Hysteresis thresholding to reduce non-edge parts
    final_img = Hysteresis_thresholding(Non_maximum_img, min_threshold, max_threshold)
    
    return Gaussian_Derive_imgx, Gaussian_Derive_imgy, Gaussian_Derive_img, Non_maximum_img, final_img


