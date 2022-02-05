'''
Author: zhuqipeng
Date: 2021-10-25 15:53:25
version: 3.5
LastEditTime: 2021-10-25 15:53:27
LastEditors: zhuqipeng
Description: 
FilePath: \RANSAC\model\eval.py
'''
import numpy as np

def eval(param, points, threshold_inlier):
    """
    return the percentage of inliers in all points 

    Parameters
    ----------
    param : tuple
            (a, b, d) for linear model
    points : array_like
            the total points you need to calculate
    threshold_inlier : float
            the threshold of discriminating whether the point is the inlier
    
    Returns
    -------
    accuracy : float
            the percentage of inliers in all points 
    """
    a, b, d = param
    distance = abs(d - np.sum(np.array([a, b]) * points, axis=1))
    correct_num = np.sum(distance < threshold_inlier, dtype=np.float32) 
    accuracy = correct_num / len(points)
    return accuracy