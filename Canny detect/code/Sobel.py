'''
Author: zhuqipeng
Date: 2021-09-04 17:05:28
version: 3.5
LastEditTime: 2021-09-04 17:39:07
LastEditors: zhuqipeng
Description: 
FilePath: \Canny detect\code\Sobel.py
'''

from Gaussian_Kernel import Gaussian_Smooth
import numpy as np
from conv import conv
from Non_maximum import Non_maximum
from Hysteresis_thresholding import Hysteresis_thresholding

def Sobel(img):
    '''
    description: Extract the edge of the picture
    @params:{np.array[[]]} a raw img
    return:{np.array[[]]} a new img after Gaussion filtering
    '''

    Gx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    Gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    img_x = conv(img, Gx)
    img_y = conv(img, Gy)
    
    new_img = np.sqrt(img_x**2 + img_y**2)
    new_img.clip(0, 255)
    new_img = np.rint(new_img).astype('uint8')

    img_x = np.rint(img_x.clip(0, 255)).astype('uint8')
    img_y = np.rint(img_y.clip(0, 255)).astype('uint8')
    

    return img_x, img_y, new_img

def Sobel_detect(img, sigma, kernelsize, min_threshold = 0.1, max_threshold = 0.3):
    Gaussian_Smooth_img = Gaussian_Smooth(img, sigma, kernelsize)
    Sobel_img_x, Sobel_img_img_y, Sobel_img = Sobel(Gaussian_Smooth_img)
    dir = np.zeros(Sobel_img_x.shape) #initialize the direction array
    for i in range(Sobel_img_x.shape[0]):
        for j in range(Sobel_img_x.shape[1]):
            dir[i, j] = np.arctan(Sobel_img_img_y[i, j] // Sobel_img_x[i, j])
    #refine the image
    Non_maximum_img = Non_maximum(Sobel_img, dir)

    #use Hysteresis thresholding to reduce non-edge parts
    final_img = Hysteresis_thresholding(Non_maximum_img, min_threshold, max_threshold)
    return Sobel_img_x, Sobel_img_img_y, Sobel_img, Non_maximum_img, final_img