'''
Author: zhuqipeng
Date: 2021-10-24 15:47:24
version: 3.5
LastEditTime: 2021-10-30 12:39:14
LastEditors: zhuqipeng
Description: 
FilePath: \RANSAC\test.py
'''

from utils.gen_data import gen_data, draw
import math
import time
import sys
sys.path.append('./model')
from model.RANSAC import RANSAC
import matplotlib.pyplot as plt
from config import config as config_
from model_selection import Adaptive_model_selection

def cal_avg_distance(points, a, b, d, threshold_inlier):
    """
    return average distance of points to the line model.

    Parameter
    ---------
    points : array like
            [[x1,y1],[x2,y2],...]
    a : float
    b : float
    d : float
    thereshold_inlier :  float
            the threshold of discriminating whether the point is the inlier

    Return
    ------
    avg_dis : float
            average distance
    inlier_rate : float
            inliers rate
    """ 
    dis_sum = 0
    inlier_num = 0
    point_num = len(points)
    for point in points:
        dis = (a * point[0] + b * point[1] - d) ** 2
        dis_sum += dis 

        if dis < threshold_inlier:
            inlier_num += 1

    avg_dis = dis_sum / point_num
    inlier_rate = inlier_num / point_num
    return avg_dis, inlier_rate

def test_gen_data():
    # test gen_data
    print("="*40, "TEST GEN_DATA", "="*40)
    start = time.time()
    test_points_num = 10000
    test_noise_rate = [0.1, 0.2, 0.3]
    test_k = -3.6
    test_b = 2.1
    test_noise_type = ['Gaussian', 'Uniform']

    for noise_type in test_noise_type:
        for noise_rate in test_noise_rate:
            print("-"*40, 'TEST PARAMS', "-"*40)
            print("test_points_num = %d,\ntest_noise_rate = %.2f,\ntest_k=%.2f,\ntest_b=%.2f,\ntest_noise_type=%s"\
                % (test_points_num, noise_rate, test_k, test_b, noise_type))
            print("-"*40, 'TEST PARAMS', "-"*40,'\n')
            points, noise_index, nonnoise_index = gen_data(test_k, test_b, test_points_num, noise_rate, noise_type)
            draw(points, noise_index, nonnoise_index, test_k, test_b)
            plt.title("test_points_num = %d, test_noise_rate = %.2f,\ntest_k=%.2f, test_b=%.2f, test_noise_type=%s"\
                % (test_points_num, noise_rate, test_k, test_b, noise_type))
            plt.savefig('./result/test_gen_data/test_gen_data_(%.2f, %s).png' % (noise_rate, noise_type))
            plt.close()
    print('running time:', time.time() - start)
    print("="*40, "TEST GEN_DATA", "="*40,'\n')

def test_adaptive():
    print("="*40, "TEST ADAPTIVE", "="*40)
    config = config_()
    points, _, __ = gen_data(config.true_k, config.true_b, config.num_of_points, config.noise_rate)
    test_min_sample_num = [10, 20, 30, 40, 50]
    for min_sample_num in test_min_sample_num:
        start = time.time()
        N_sum = 0
        max_iters = Adaptive_model_selection(points, config.num_of_points, config.threshold_inlier, min_sample_num)
        
        print('min sample num: ', min_sample_num)
        print('Max iter: ', max_iters)
        print('adaptive select max iter for 100 times time:%.2f sec\n' % (time.time() - start))
    print("="*40, "TEST ADAPTIVE", "="*40,'\n')

def test_RANSAC(true_k, true_b, num_of_points, noise_rate, noise_type, max_iters, threshold_model, threshold_inlier):
    points, noise_index, nonnoise_index = gen_data(true_k, true_b, num_of_points, noise_rate, noise_type)
    [a_hat, b_hat, d_hat] = RANSAC(points, max_iters = max_iters, num_of_points=num_of_points, threshold_inlier=num_of_points, threshold_model=threshold_model)
    draw(points, noise_index, nonnoise_index, true_k, true_b, True, -a_hat/b_hat, d_hat/b_hat)
    avg_dis_hat,  inlier_rate_hat = cal_avg_distance(points, a_hat, b_hat, d_hat, threshold_inlier)

    
    true_a = -true_k * (1 /math.sqrt(true_k ** 2 + 1))
    true_d = true_b * (1 /math.sqrt(true_k ** 2 + 1))
    true_b = 1 /math.sqrt(true_k ** 2 + 1)

    avg_dis_true, inlier_rate_true = cal_avg_distance(points, true_a, true_b, true_d, threshold_inlier)

    print('true_a:', true_a, '\tfitting_a:', a_hat)
    print('true_b:', true_b, '\tfitting_b:', b_hat)
    print('true_d:', true_d, '\tfitting_d:', d_hat)
    print('inlier_rate_true:', inlier_rate_true, '\tavg distance_true:', avg_dis_true)
    print('inlier_rate_hat:', inlier_rate_hat, '\tavg distance_hat:', avg_dis_hat)

def test_RANSAC_noise_rate(test_noise_type, test_noise_rate, config):
    print('-'*40,'Test test_noise_rate','-'*40)
    for noise_type in test_noise_type:
        for noise_rate in test_noise_rate:
            start = time.time()
            print("-"*40, 'TEST PARAMS', "-"*40)
            print("true_k=%.2f, \ntrue_b=%.2f, \nnum_of_points=%d, \nnoise_rate=%.2f, \nnoise_type=%s, \nmax_iters=%d, \nthreshold_model=%.2f, \nthreshold_inlier=%.2f"\
                % (config.true_k, config.true_b, config.num_of_points, noise_rate, noise_type, config.max_iters, config.threshold_model, config.threshold_inlier))
            print("-"*40, 'TEST PARAMS', "-"*40,'\n')

            test_RANSAC(config.true_k, config.true_b, config.num_of_points, noise_rate, noise_type, config.max_iters, config.threshold_model, config.threshold_inlier)

            plt.title("true_k=%.2f, true_b=%.2f, num_of_points=%d, noise_rate=%.2f, \nnoise_type=%s, max_iters=%d, threshold_model=%.2f, threshold_inlier=%.2f"\
                % (config.true_k, config.true_b, config.num_of_points, noise_rate, noise_type, config.max_iters, config.threshold_model, config.threshold_inlier))
            plt.savefig('./result/test_RANSAC_Param/test_noise_rate/test_noise_rate_(%.2f, %s).png' % (noise_rate, noise_type))
            plt.close()
            print('Running time: %.2f sec' % (time.time() - start))
    print('-'*40,'Test test_noise_rate','-'*40)

def test_RANSAC_threshold_inlier(test_noise_type, test_threshold_inlier, config):
    print('-'*40,'Test test_threshold_inlier','-'*40)
    for noise_type in test_noise_type:
        for threshold_inlier in test_threshold_inlier:
            start = time.time()
            print("-"*40, 'TEST PARAMS', "-"*40)
            print("true_k=%.2f, \ntrue_b=%.2f, \nnum_of_points=%d, \nnoise_rate=%.2f, \nnoise_type=%s, \nmax_iters=%d, \nthreshold_model=%.2f, \nthreshold_inlier=%.2f"\
                % (config.true_k, config.true_b, config.num_of_points, config.noise_rate, noise_type, config.max_iters, config.threshold_model, threshold_inlier))
            print("-"*40, 'TEST PARAMS', "-"*40,'\n')

            test_RANSAC(config.true_k, config.true_b, config.num_of_points, config.noise_rate, noise_type, config.max_iters, config.threshold_model, threshold_inlier)

            plt.title("true_k=%.2f, true_b=%.2f, num_of_points=%d, noise_rate=%.2f, \nnoise_type=%s, max_iters=%d, threshold_model=%.2f, threshold_inlier=%.2f"\
                % (config.true_k, config.true_b, config.num_of_points, config.noise_rate, noise_type, config.max_iters, config.threshold_model, threshold_inlier))
            plt.savefig('./result/test_RANSAC_Param/test_threshold_inlier/test_threshold_inlier_(%.2f, %s).png' % (threshold_inlier, noise_type))
            plt.close()
            print('Running time: %.2f sec' % (time.time() - start))
    print('-'*40,'Test test_noise_rate','-'*40)

def test_RANSAC_threshold_model(test_noise_type, test_threshold_rate_model, config):
    print('-'*40,'Test test_threshold_model','-'*40)
    for noise_type in test_noise_type:
        for threshold_model_rate in test_threshold_rate_model:
            start = time.time()
            threshold_model = int(threshold_model_rate * config.num_of_points)
            print("-"*40, 'TEST PARAMS', "-"*40)
            print("true_k=%.2f, \ntrue_b=%.2f, \nnum_of_points=%d, \nnoise_rate=%.2f, \nnoise_type=%s, \nmax_iters=%d, \nthreshold_model=%.2f, \nthreshold_inlier=%.2f"\
                % (config.true_k, config.true_b, config.num_of_points, threshold_model, noise_type, config.max_iters, threshold_model, config.threshold_inlier))
            print("-"*40, 'TEST PARAMS', "-"*40,'\n')

            test_RANSAC(config.true_k, config.true_b, config.num_of_points, config.noise_rate, noise_type, config.max_iters, threshold_model, config.threshold_inlier)
            plt.title("true_k=%.2f, true_b=%.2f, num_of_points=%d, noise_rate=%.2f, \nnoise_type=%s, max_iters=%d, threshold_model=%.2f, threshold_inlier=%.2f"\
                % (config.true_k, config.true_b, config.num_of_points, config.noise_rate, noise_type, config.max_iters, threshold_model, config.threshold_inlier))
            plt.savefig('./result/test_RANSAC_Param/test_threshold_model/test_threshold_model_(%.2f, %s).png' % (threshold_model, noise_type))
            plt.close()
            print('Running time: %.2f sec' % (time.time() - start))
    print('-'*40,'Test test_threshold_model','-'*40)

def test_RANSAC_Params():
    print("="*40, "TEST RANSAC", "="*40)
    config = config_()
    test_noise_rate = [0.1, 0.2, 0.3]
    test_noise_type = ['Gaussian', 'Uniform']
    test_threshold_inlier = [0.1, 0.3, 0.5]
    test_threshold_rate_model = [0.75, 0.85, 0.95]
    test_RANSAC_noise_rate(test_noise_type, test_noise_rate, config)
    test_RANSAC_threshold_inlier(test_noise_type, test_threshold_inlier, config)
    test_RANSAC_threshold_model(test_noise_type, test_threshold_rate_model, config)
    print("="*40, "TEST RANSAC", "="*40)

def main():
    config = config_()
    points, noise_index, nonnoise_index = gen_data(config.true_k, config.true_b, config.num_of_points, config.noise_rate, 'Uniform')
    draw(points, noise_index, nonnoise_index, config.true_k, config.true_b)

    [a_hat, b_hat, d_hat] = RANSAC(points, max_iters = config.max_iters, num_of_points=config.num_of_points, threshold_inlier=config.threshold_inlier, threshold_model=config.threshold_model)
    avg_dis_hat,  inlier_rate_hat = cal_avg_distance(points, a_hat, b_hat, d_hat, config.threshold_inlier)

    true_b = 1 /math.sqrt(config.true_k ** 2 + 1)
    true_a = -config.true_k * true_b
    true_d = config.true_b * true_b

    avg_dis_true, inlier_rate_true = cal_avg_distance(points, true_a, true_b, true_d, config.threshold_inlier)

    print('true_a:', true_a, '\tfitting_a:', a_hat)
    print('true_b:', true_b, '\tfitting_b:', b_hat)
    print('true_b:', true_d, '\tfitting_b:', d_hat)
    print('inlier_rate_true:', inlier_rate_true, '\tavg distance_true:', avg_dis_true)
    print('inlier_rate_hat:', inlier_rate_hat, '\tavg distance_hat:', avg_dis_hat)

if __name__ == '__main__':
    test_adaptive()






