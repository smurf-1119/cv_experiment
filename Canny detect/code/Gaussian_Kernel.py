'''
Author: zhuqipeng
Date: 2021-09-04 17:01:56
version: 3.5
LastEditTime: 2021-09-04 17:01:57
LastEditors: zhuqipeng
Description: 
FilePath: \Canny detect\Gaussian_Kernel.py
'''
import numpy as np
from conv import conv

def Gaussian_function(sigma, x, y):
    '''
    description: 2-dim Gaussian function
    @params: {float} sigma
    @params: {float} x
    @params: {float} y
    return: np.exp(- ((x**2 + y**2) / (2 * sigma**2))) / (2 * np.pi * sigma**2)
    '''
    return np.exp(- ((x**2 + y**2) / (2 * sigma**2))) / (2 * np.pi * sigma**2)

def Get_Gaussian_Kernel(sigma=1.3, kernelsize = 3):
    '''
    description: Get Gaussian Kernel with the sigma and kernelsize
    @param:{int} kernelsize
    return: A Gaussian Kernel with kernel size
    '''
    center_pos = kernelsize // 2 #get center position of kernel
    Gaussian_Kernel = np.zeros((kernelsize, kernelsize))# initialize Gaussian Kernel

    for i in range(center_pos + 1):
        for j in range(center_pos + 1):
            #can use the symmetry of 2-dim Gaussian function
            Gaussian_Kernel[i,j] = Gaussian_function(sigma, i - center_pos, j - center_pos)
            Gaussian_Kernel[2*center_pos - i, j] = Gaussian_Kernel[i,j]
            Gaussian_Kernel[i, 2*center_pos - j] = Gaussian_Kernel[i,j]
            Gaussian_Kernel[2*center_pos - i, 2*center_pos - j] = Gaussian_Kernel[i,j]
    
    Gaussian_Kernel = Gaussian_Kernel / np.sum(Gaussian_Kernel) #Normalization 归一
    return Gaussian_Kernel

def Gaussian_Smooth(img, sigma, kernelsize):
    Gaussian_Kernel = Get_Gaussian_Kernel(sigma, kernelsize)
    new_img = conv(img, Gaussian_Kernel)
    new_img = np.rint(new_img.clip(0, 255)).astype('uint8')
    return new_img

def Get_Gaussian_Derive_x_Kernel(sigma, kernel_size = 3):
    '''
    description
    @param:{int} kernel_size
    return: Gaussian_Derive_x_Kernel
    '''

    #get the Gaussian Derive Kernel with the kernel size
    #then use [1, 0, -1] to derive the Gaussian Kenel
    Gaussian_kernel = Get_Gaussian_Kernel(sigma, kernel_size)
    Gaussian_Derive_x_Kernel = conv(Gaussian_kernel, np.array([[1, 0, -1]]))
    
    return Gaussian_Derive_x_Kernel

def Get_Gaussian_Derive_y_Kernel(sigma, kernelsize = 3):
    '''
    description
    @param:{int} kernel_size
    return: Gaussian_Derive_y_Kernel
    '''

    #get the Gaussian Derive Kernel with the kernel size
    #then use [[1], [0], [-1]] to derive the Gaussian Kenel
    Gausian_kernel = Get_Gaussian_Kernel(sigma, kernelsize)
    Gaussian_Derive_y_Kernel = conv(Gausian_kernel, np.array([[1], [0], [-1]]))

    return Gaussian_Derive_y_Kernel

def Gaussian_Derive_Filter(img, sigma, kernelsize):
    '''
    description: use Gaussian Derive kernel to filter the img
    @params: {array[[]]} img
    @params: {float} sigma
    @params: {float} kernelsize
    return: img_x, img_y, result_img, dir
    '''

    #Get Gaussian_Derive_x_Kernel and Gaussian_Derive_y_Kernel separately
    Gx = Get_Gaussian_Derive_x_Kernel(sigma, kernelsize)
    Gy = Get_Gaussian_Derive_y_Kernel(sigma, kernelsize)
    #use the kernel to convolute images separately
    img_x = conv(img, Gx)
    img_y = conv(img, Gy)
    
    result_img = np.sqrt(img_x**2 + img_y**2)
    result_img.clip(0, 255) #control the pixel value from 0 to 255
    result_img = np.rint(result_img).astype('uint8') #convert pixels of result_img value to integer
    img_x = np.rint(img_x.clip(0, 255)).astype('uint8')
    img_y = np.rint(img_y.clip(0, 255)).astype('uint8')

    dir = np.zeros(img_x.shape) #initialize the direction array
    for i in range(img_x.shape[0]):
        for j in range(img_x.shape[1]):
            dir[i, j] = np.arctan(img_y[i, j] // img_x[i, j])
 

    return img_x, img_y, result_img, dir
