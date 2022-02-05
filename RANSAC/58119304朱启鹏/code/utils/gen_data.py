import numpy as np
from IPython import display
import matplotlib.pyplot as plt


def gen_data(true_k, true_b, num_of_points, noise_rate, noise_type='Gaussian'):
    """
    return the generated data

    Parameters
    ----------
    num_of_points : int
                    the number of generating points
    noise_rate : float
                    the percentage of noise points in all points
    noise_type : str
                    the type of noise -> 'Gaussian' of 'Uniform', default 'Gaussian'
    
    Returns
    -------
    points : ndarray
                    point array, [[x1,y1]. [x2,y2], ...]
    noise_index : ndarray
                    record the indexs of noise points
    nonnoise_index : ndarry
                    record the indexs of nonnoise points
    """
    #generate all points 
    noise_points_num = int(num_of_points * noise_rate)
    x = np.random.randn(num_of_points, 1)
    y = true_k * x + true_b
    y += np.random.normal(0, 0.3, size=(num_of_points,1))

    #generate noise points
    rand_list = np.arange(num_of_points)
    np.random.shuffle(rand_list)
    noise_index = rand_list[:noise_points_num]
    nonnoise_index = rand_list[noise_points_num:]
    if noise_type == 'Gaussian':
        y[noise_index] = np.random.normal(0, 4, size=(noise_points_num,1))
    elif noise_type == 'Uniform':
        y[noise_index] = np.random.uniform(min(y), max(y), size=(noise_points_num,1))
    else:
        raise NameError

    points = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1)
    return points, noise_index, nonnoise_index

def draw(points, noise_index, nonnoise_index, k, b, draw_test_line=False, k_hat=None, b_hat=None):
    '''

    Parameter
    ---------
    points : array like
    noise_index : ndarray
                  record the indexs of noise points
    nonnoise_index : ndarry
                  record the indexs of nonnoise points
    k : float
    b : float 
    
    '''
    def use_svg_display():
        # 用矢量图显示
        display.set_matplotlib_formats('svg')

    def set_figsize(figsize=(10, 10)):
        use_svg_display()
        # 设置图的尺寸
        plt.rcParams['figure.figsize'] = figsize

    x = points[:,0]
    y = x * k + b

    set_figsize()
    plt.scatter(points[nonnoise_index, 0], points[nonnoise_index, 1], color='c', marker='.',label='Inliers')
    plt.scatter(points[noise_index, 0], points[noise_index, 1], color='r', marker='*', label='Outliers')
    plt.plot(x, y, color='y', linewidth=2, label='Line Model')

    if draw_test_line:
        y_hat = x * k_hat + b_hat 
        plt.plot(x, y_hat, color='gold', linewidth=2, label='Fitting Line Model')

    plt.legend(loc='upper left')
    plt.xlabel("input")
    plt.ylabel("response")