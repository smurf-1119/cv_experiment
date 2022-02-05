'''
Author: zhuqipeng
Date: 2021-10-25 15:47:51
version: 3.5
LastEditTime: 2021-10-28 18:59:56
LastEditors: zhuqipeng
Description: 
FilePath: \RANSAC\model_selection.py
'''
from IPython import start_ipython
from utils.gen_data import gen_data
import math
import sys
import time
sys.path.append('./model')
from model.RANSAC import RANSAC_Each_iter 
from config import config as config_


def Adaptive_model_selection(points, num_of_points, threshold_inlier, min_sample_num, p=0.99):
    """
    return the best max_iters

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
    p : float
        the probability of condecting max_iters times experiment of getting inliers all the time
    
    Returns
    -------
    max_iters : float
        the best number of iter

    """
    max_iters = 99999
    sample_count = 0
    num_of_points = len(points)
    outlier_ratio = 1

    while max_iters > sample_count:
        inliers_list = RANSAC_Each_iter(points, num_of_points, threshold_inlier, min_sample_num=2)
        num_of_inliers = len(inliers_list)

        outlier_ratio_new = 1 - (num_of_inliers) / (num_of_points)
        if outlier_ratio_new < outlier_ratio:
            outlier_ratio = outlier_ratio_new
            try:
                max_iters = int(math.log(1-p) / math.log(1 - math.pow(1-outlier_ratio,min_sample_num)))
            except ZeroDivisionError:
                pass
            
        sample_count += 1
    

    return max_iters

def save_params(params, file_path='config.yaml'):
    import yaml
    with open(file_path, 'w') as f:
        yaml.dump(params, f)
    f.close()

def main():
    config = config_()
    points, _, __ = gen_data(config.true_k, config.true_b, config.num_of_points, config.noise_rate)
    N_sum = 0
    for _ in range(100):
        N = Adaptive_model_selection(points, config.num_of_points, config.threshold_inlier, config.min_sample_num)
        N_sum += N
    max_iters = N_sum / 100

    params = \
    {
        'max_iters': max_iters,
        'min_sample_num': config.min_sample_num,
        'noise_rate': config.noise_rate,
        'num_of_points': config.num_of_points,
        'threshold_inlier': config.threshold_inlier,
        'threshold_model': config.threshold_inlier,
        'true_b': config.true_b,
        'true_k': config.true_k,
    }

    save_params(params)

if __name__ == '__main__':
    main()


