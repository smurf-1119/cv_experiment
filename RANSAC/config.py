'''
Author: zhuqipeng
Date: 2021-10-25 15:55:22
version: 3.5
LastEditTime: 2021-10-25 16:26:56
LastEditors: zhuqipeng
Description: 
FilePath: \RANSAC\config.py
'''
import yaml

class config():
    def __init__(self, file_path='config.yaml') -> None:
        params = read_parameters(file_path)
        self.max_iters = params['max_iters']
        self.min_sample_num = params['min_sample_num']
        self.noise_rate = params['noise_rate']
        self.num_of_points = params['num_of_points']
        self.threshold_inlier = params['threshold_inlier']
        self.threshold_model = params['threshold_model']
        self.true_b = params['true_b']
        self.true_k = params['true_k']

def read_parameters(file_path='config.yaml'):
    # 读
    # 用open方法打开直接读取
    with open(file_path, 'r') as f:
        params = f.read()
    dict_params = yaml.load(params, Loader=yaml.FullLoader) # 用load方法转字典 d = yaml.load_all(cfg)
    return dict_params        
