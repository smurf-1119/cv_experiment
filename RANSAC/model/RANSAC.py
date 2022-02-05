'''
Author: zhuqipeng
Date: 2021-10-25 15:50:15
version: 3.5
LastEditTime: 2021-10-28 16:58:57
LastEditors: zhuqipeng
Description: 
FilePath: \RANSAC\model\RANSAC.py
'''

import numpy as np
import time
from eval import eval

def model_fit(points):
    """
    return the linear model fitting the given points

    Parameters
    ----------
    points : array_like
            all points you need to calculate

    Returns
    -------
    a, b, d : float
            the parameters of the fitting model
    """
    # if len(points) == 2:
    #     k = (points[0, 1] - points[1, 1]) / (points[0, 0] - points[1, 0])
    #     a = - k / math.sqrt(k**2 + 1)
    #     b = 1 / math.sqrt(k**2 + 1)
    #     d = a * points[0, 0] + b * points[0, 1]
    #     return np.array([a, b]), d
    # else:
    #     mean = np.mean(points, axis=0)
    #     U = points - mean
    #     A = np.matmul(U.T, U)
    #     eig_values, eig_vecs=np.linalg.eig(A)
    #     N = eig_vecs[:,np.argmin(eig_values)].squeeze()
    #     d = np.sum(N*mean)
    #     return N, d
    mean = np.mean(points, axis=0)
    U = points - mean
    A = np.matmul(U.T, U)
    eig_values, eig_vecs=np.linalg.eig(A)
    a, b = eig_vecs[:,np.argmin(eig_values)].squeeze()
    d = a * mean[0] + b * mean[1]
    return a, b, d

def RANSAC_Each_iter(points, num_of_points, threshold_inlier, min_sample_num):
    """
    return the best linear model by using RANSAC algorithm

    Parameters
    ----------
    points: array_like
        point array, [[x1,y1]. [x2,y2], ...]
    num_of_points : int
        total numbers of points
    threshold_inlier : int
        the threshold of discriminating whether the point is the inlier
    min_sample_num : int
        the size 0f subset, defalt 2
    
    Returns
    -------
    inliers_list : ndarray
        the list of inliers

    """

    #draw the sample points as a number of min_sample_num
    rand_list = np.arange(num_of_points)
    np.random.shuffle(rand_list)
    sample_points = points[rand_list[:min_sample_num], :]

    #fit the model using the sample points
    a, b, d = model_fit(sample_points)

    #get all points except sample points 
    nonsample_x = points[:,0][rand_list[min_sample_num:]]
    nonsample_y = points[:,1][rand_list[min_sample_num:]]

    #calculate the numbers of inliers of the lastest fitting model
    inliers_list = []
    for (u, v) in zip(nonsample_x, nonsample_y):
        distance = (a * u + b * v - d) ** 2
        if distance < threshold_inlier:
            inliers_list.append(np.array([u, v]).squeeze())
    return inliers_list

def RANSAC(points, max_iters, num_of_points, threshold_inlier, threshold_model, min_sample_num=2):
    """
    return the best linear model by using RANSAC algorithm

    Parameters
    ----------
    points: array_like
        point array, [[x1,y1]. [x2,y2], ...]
    max_iters : int
        max iterations
    num_of_points : int
        total numbers of points
    threshold_inlier : float
        the threshold of discriminating whether the point is the inlier
    threshold_model : int
        the threshold of discriminating whether the model is good
    min_sample_num : int
        the size 0f subset, defalt 2
    
    Returns
    -------
    params : ndarray
        [a, b, d] for the best linear model
    """
    
    #initialize
    start = time.time()
    params = []
    num_of_points = len(points)

    #iterate maxiters times
    for i in range(max_iters):
        inliers_list = RANSAC_Each_iter(points, num_of_points, threshold_inlier, min_sample_num)

        #discreminate whether the model is good, and fit its all inliers to get its new params
        if len(inliers_list) >= threshold_model:
            a, b, d = model_fit(np.array(inliers_list))
            params.append([a, b, d])

    #evaluate all equalified models, and return the best model
    best_acc = 0
    best_model_index = 0
    for i, (a, b, d) in enumerate(params):
        acc = eval((a, b, d), points, threshold_inlier)
        if acc > best_acc:
            best_acc = acc
            best_model_index = i
    
    RANSAC_time = time.time() - start
    print('Running %.2f sec'%(RANSAC_time))
    print('best_acc: ', best_acc)
    return params[best_model_index]