# <center>Canny Detect Projecr<center>

朱启鹏 58119304

## 1. 实验内容

### 1.1 实验目标

- 实现对彩色图像的灰度处理，并使用高斯一阶导数滤波器计算图像梯度，进而执行非极大值抑制和阈值操作及连接，从而进行canny边缘检测。

- 具体实现任务：
  - 原图按灰度通道读取
  - 实现图像卷积函数
  - 高斯一阶偏导滤波实现
  - 非极大值抑制
  - 双边阈值

### 1.2 实验数据集

- 实验用图：
  - 数字图像处理经典图片`Lena`等。

### 1.3 编译环境

- python 3.7

## 2 实验思路

### 2.1 二维图像卷积

- 公式：

$$
f[n, m] * h[n, m]=\sum_{k=-\infty}^{\infty} \sum_{l=-\infty}^{\infty} f[k, l] h[n-k, m-l]
$$

- Strading:(灰色部分为补零位置，白色部分为原图)

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109021636379.png" alt="image-20210902163607174" style="zoom:80%;" />

- Convolution:

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109021637456.png" alt="image-20210902163732028" style="zoom: 67%;" />

- 让Kernel的中心遍历图像的每一个像素。

### 2.2 高斯一阶偏导滤波

### 2.2.1 Gaussian Kernel 实现

- 以核的中心为原点建立直角坐标系，利用二维高斯函数，计算图片上每一个像素值。
- 2-dim Gaussian Function:

$$
G_{\sigma}=\frac{1}{2 \pi \sigma^{2}} e^{-\frac{\left(x^{2}+y^{2}\right)}{2 \sigma^{2}}}
$$

- 对kernel进行归一化处理

### 2.2.2 Gaussian Derive Kernel 实现

- 对得到的Gaussian Kernel分别在x方向和y方向求偏导
- 具体而言，即分别使用$[1,0,-1]$与$[1,0,-1]^t$​​​分别高斯核进行卷积处理，从而分别得到x方向的高斯偏导核`Gx`以及y方向的高斯偏导核`Gy`

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109021758695.png" alt="image-20210902175847450" style="zoom: 67%;" />

- 分别使用`Gx`以及`Gy`对原图进行卷积，从而分别得到x方向核y方向的高斯偏导滤波图`img_Gx`和`img_Gy`
- 特别的，处理完的图片需要将小于零的像素归为0，大于255的像素归为255，并将像素值转化为整型
- 接着再利用模值公式以及方向公式分别计算出`img_G`和`dir`

$$
\nabla f=\left[\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right]
$$


$$
\|\nabla f\|=\sqrt{\left(\frac{\partial f}{\partial x}\right)^{2}+\left(\frac{\partial f}{\partial y}\right)^{2}}
$$

$$
\theta=\tan ^{-1}\left(\frac{\partial f}{\partial y} / \frac{\partial f}{\partial x}\right)
$$

- 其中，将模值作为此次滤波的结果图片

### 2.3 非极大值抑制

- 非极大值抑制是一种边缘稀疏技术，非极大值抑制的作用在于“瘦”边。对图像进行梯度计算后，仅仅基于梯度值提取的边缘仍然很模糊。对于标准3，对边缘有且应当只有一个准确的响应。而非极大值抑制则可以帮助将局部最大值之外的所有梯度值抑制为0，对梯度图像中每个像素进行非极大值抑制的算法是：
  - 将当前像素的梯度强度与沿正负梯度方向上的两个像素进行比较。
  - 如果当前像素的梯度强度与另外两个像素相比最大，则该像素点保留为边缘点，否则该像素点将被抑制。
- 通常为了更加精确的计算，在跨越梯度方向的两个相邻像素之间使用线性插值来得到要比较的像素梯度，现举例如下：

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031024806.png" alt="image-20210903102405726" style="zoom:80%;" />

- 如图所示，将梯度分为8个方向，分别为E、NE、N、NW、W、SW、S、SE，其中0代表0°~45°或者-180°~-135°,1代表45°~90°或者-135°~-90°，2代表90°~135°或者-90°~-45°，3代表135°~180°或者45°~0°。如图，像素点P的梯度方向为$\theta$​​，则像素点P1和P2的梯度线性插值为： 

- 0区：
  - 特别的，当 $\theta<0$​ 时，$\theta:=\theta+\pi$​

$$
G_{p 1}=(1-\tan (\theta)) \times E+\tan (\theta) \times N E
$$

$$
G_{p 2}=(1-\tan (\theta)) \times W+\tan (\theta) \times S W
$$

- 1区：
  - 特别的，当 $\theta<0$​​​ 时，$\theta:=-\theta-\frac{\pi}{2}$​​​ ; 当 $\theta\ge0$​​​ 时，$\theta:=-\theta+\frac{\pi}{2}$​​​ ; 

$$
G_{p 1}=(1-\tan (\theta)) \times N+\tan (\theta) \times N E
$$

$$
G_{p 2}=(1-\tan (\theta)) \times S+\tan (\theta) \times S W
$$

- 2区：
  - 特别的，当 $\theta<0$​​​ 时，$\theta:=\theta+\frac{\pi}{2}$​​​ ; 当 $\theta\ge0$​​​ 时，$\theta:=\theta-\frac{\pi}{2}$​​​ ; 

$$
G_{p 1}=(1-\tan (\theta)) \times N+\tan (\theta) \times N W
$$

$$
G_{p 2}=(1-\tan (\theta)) \times S+\tan (\theta) \times S E
$$
- 3区：
  - 特别的，当 $\theta<0$​​​​​ 时，$\theta:=-\theta+\pi$​ ; 当 $\theta\ge0$​ 时，$\theta:=-\theta$​​​​​ ; 

$$
G_{p 1}=(1-\tan (\theta)) \times W+\tan (\theta) \times N W
$$

$$
G_{p 2}=(1-\tan (\theta)) \times E+\tan (\theta) \times S E
$$
- 因此非极大值抑制的伪代码描写如下：

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031040622.png" alt="image-20210903104027306" style="zoom:80%;" />

### 2.4 双阈值检测

- 在施加非极大值抑制之后，剩余的像素可以更准确地表示图像中的实际边缘。然而，仍然存在由于噪声和颜色变化引起的一些边缘像素。为了解决这些杂散响应，必须用弱梯度值过滤边缘像素，并保留具有高梯度值的边缘像素，可以通过选择高低阈值来实现。如果边缘像素的梯度值高于高阈值，则将其标记为强边缘像素；如果边缘像素的梯度值小于高阈值并且大于低阈值，则将其标记为弱边缘像素；如果边缘像素的梯度值小于低阈值，则会被抑制。阈值的选择取决于给定输入图像的内容。
- 双阈值检测的伪代码描写如下：

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031041933.png" alt="image-20210903104147416" style="zoom:80%;" />

### 2.5 抑制孤立低阈值点

- 到目前为止，被划分为强边缘的像素点已经被确定为边缘，因为它们是从图像中的真实边缘中提取出来的。然而，对于弱边缘像素，将会有一些争论，因为这些像素可以从真实边缘提取也可以是因噪声或颜色变化引起的。为了获得准确的结果，应该抑制由后者引起的弱边缘。通常，由真实边缘引起的弱边缘像素将连接到强边缘像素，而噪声响应未连接。为了跟踪边缘连接，通过查看弱边缘像素及其8个邻域像素，只要其中一个为强边缘像素，则该弱边缘点就可以保留为真实的边缘。
- 特别的，当出现弱边缘像素周围全为弱边缘像素，则无法判定，则先把该像素放入队列中
- 每次迭代结束，都判断一下是否有新的弱边缘变为强边缘，如果没有则跳出循环；否则，则将队列1赋值为队列2
- 抑制孤立边缘点的伪代码描述如下：

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031101037.png" alt="image-20210903110107851" style="zoom:67%;" />

## 3. 实验过程

### 3.1 原图按灰度通道读取

- 本次读取使用`cv2.imread`函数读取图片，并用`plt.show()`展示，效果如下：

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109021604680.png" alt="image-20210902160446082" style="zoom:80%;" />

### 3.2 高斯一阶偏导滤波

#### 3.2.1 高斯核生成

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109021744109.png" alt="image-20210902174415319" style="zoom: 67%;" />

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109021744521.png" alt="image-20210902174437550" style="zoom:80%;" />

- 我们可以发现该高斯核呈现类似高斯函数的分布，中间数值高，周围数值低，所以实验成功。

#### 3.2.2 高斯偏导核生成

##### 3.2.2.1 x方向

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031116745.png" alt="image-20210903111641745" style="zoom:67%;" />

<img src="E:/third_year_in_University/CV/experiment/Canny%20detect/img/image-20210903111849848.png" alt="image-20210903111849848" style="zoom:80%;" />

- 我们可以发现该高斯核在x方向呈现类似高斯x方向偏导函数的分布，所以实验成功。

##### 3.2.2.2 y方向

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031119078.png" alt="image-20210903111923244" style="zoom:67%;" />

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031119197.png" alt="image-20210903111956669" style="zoom:80%;" />

- 我们可以发现该高斯核在y方向呈现类似高斯y方向偏导函数的分布，所以实验成功。

##### 3.2.2.3 滤波

- 我们知道，$ \sigma$越小对越能提取边缘信息，但是有可能把不必要的噪音也保留；$ \sigma$越大，虽然越容易消除噪音的影响，但是可能造成边缘确实的影响，所以需要合适的$ \sigma $才能才能保证滤波成功。
- 而一般而言，kernel的大小取3$ \sigma $的最靠近奇数。
- 下图利用自行车的图片进行实验，便于寻找合适的$ \sigma$和kernel大小

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031553408.png" alt="image-20210903155314649" style="zoom: 80%;" />

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031614736.png" alt="image-20210903161447970" style="zoom: 80%;" />

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031554804.png" alt="image-20210903155428461" style="zoom: 80%;" />





<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031558912.png" alt="image-20210903155828566" style="zoom: 80%;" />

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031601694.png" alt="image-20210903160124452"  />

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031615935.png" alt="image-20210903161548133" style="zoom:80%;" />

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031616051.png" alt="image-20210903161631163" style="zoom:80%;" />

- 我们可以发现当$ \sigma$取到3时，边缘已经过粗，并没有很好的提取出边缘
- 从实验结果看，$ \sigma$​取1~1.7就可以很好的完成任务
- 所以之后实验取$ \sigma=1.3,kernel\_size=3$​
- 得到下图：

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031623848.png" alt="image-20210903162343036" style="zoom: 80%;" />

### 3.3 非极大值抑制

- 为解决边缘过粗问题，接着进行非极大值抑制。

<img src="E:/third_year_in_University/CV/experiment/Canny%20detect/img/image-20210903162425344.png" alt="image-20210903162425344" style="zoom:80%;" />

- 我们可以发现，边缘确实细化很多，但好像有点看不清，这是因为图片的对比度不够高，这在之后转换为二值图像就可以解决。

### 3.4 双边阈值

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031627937.png" alt="image-20210903162733906" style="zoom:80%;" />

- 进行双边阈值后，我们可以明显发现，图像效果好了很多，并且没有出现边缘过粗的情况。
- 另外我还分别实验了$ \sigma=1$​ ，$kernel\_size=3$​与 $ \sigma=1.7$​ ，$kernel\_size=5$​​

<img src="E:/third_year_in_University/CV/experiment/Canny%20detect/img/image-20210903162916393.png" alt="image-20210903162916393"  />

![image-20210903163219841](https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031632045.png)

- 我们可以发现 $ \sigma=1.7$ ，$kernel\_size=5$的效果最好。

## 4. 实验对比

- 与课上提过的Sobel算子，Prewitt算子与本次实验的Sobel算子进行对比

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031712877.png" alt="image-20210903171245848" style="zoom: 80%;" />

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031716795.png" alt="image-20210903171653541" style="zoom: 50%;" />

- 我们可以发现，由于没有边缘细化以及双边阈值，另外梁总算子提取的边缘都出现了过粗的现象，其中对于sobel算子面部等受到光照影响的非边缘部分也被保留，而Prewitt算子则出现了一些部位较为模糊的现象。

- 一些其他图片：

<img src="E:/third_year_in_University/CV/experiment/Canny%20detect/img/image-20210903172740746.png" alt="image-20210903172740746" style="zoom:80%;" />

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031741940.png" alt="image-20210903174150946" style="zoom:80%;" />

<img src="https://gitee.com/zhu-qipeng/blogImage/raw/master/blogImage/202109031742642.png" alt="image-20210903174221063" style="zoom:80%;" />

## 5. 代码

```python
# coding=utf-8
'''
Author: zhuqipeng
Date: 2021-08-22 22:38:51
version: 3.5
LastEditTime: 2021-09-03 17:52:00
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

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.core.fromnumeric import shape
from icecream import ic
import argparse
from mpl_toolkits import mplot3d
from matplotlib.pyplot import MultipleLocator

def set_interact_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default='./data/sample.jpg', type=str, required=False, help='输入图片路径')
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

def Filp_Img(img, flag):
    '''
    description: Flip the image, 
                if flag = 1, flip the image in vertical direction;
                if flag = 0, flip the image in horizontal direction;
                if flag = -1, flip the image in vertical and horizontal directions.
    @params: {array[[]]} img
    @params: {int} flag
    return: {array[[]]} result_img
    '''
    (row, col) = img.shape
    center = (row//2, col//2)
    new_img = np.copy(img)

    if(flag == 1):
        for i in range(row):
            for j in range(col//2):
                new_img[i, j], new_img[i, 2 * center[1] - j] = new_img[i, 2 * center[1] - j], new_img[i, j]
    elif(flag == 0):
        for i in range(row//2 + 1):
            for j in range(col):
                new_img[i, j], new_img[2 * center[0] - i, j] = new_img[2 * center[0] - i, j], new_img[i, j]
    else:
        for i in range(row//2):
            for j in range(col):
                new_img[i, j], new_img[2 * center[0] - i, 2 * center[1] - j] = new_img[2 * center[0] - i, 2 * center[1] - j], new_img[i, j]
        for j in range(col // 2):
            new_img[center[0], j], new_img[center[0], 2 * center[1] - j] = new_img[center[0], 2 * center[1] - j], new_img[center[0], j]
    return new_img

def conv(img, kernel):
    '''
    description: image convolution function, input a two-dim image and a two-dim kernel
    @params: {array[[]]} img
    @params: {array[[]]} kernel
    return: {array[[]]} result image after convolution
    '''

    (img_row, img_col) = img.shape #get the original image shape
    (kernel_row, kernel_col) = kernel.shape # get the kernel shape
    kernel_flip = Filp_Img(kernel, -1) #Flip the original image
    strading_img = np.zeros((img_row + kernel_row - 1, img_col + kernel_col - 1)) 
    strading_img[(kernel_row - 1)//2:img_row + (kernel_row - 1)//2, 
    (kernel_col - 1)//2:img_col + (kernel_col - 1)//2] = img[:, :] #initialize the strading image
    result_img = np.zeros((img_row, img_col)) #initialize the result image with zeros
    
    #Convolution
    for i in range(img_row):
        for j in range(img_col):
            result_img[i, j] = np.sum(kernel_flip * strading_img[i:i + kernel_row, j:j + kernel_col])
            
    return result_img

def Gaussian_function(sigma, x, y):
    '''
    description: 2-dim Gaussian function
    @params: {float} sigma
    @params: {float} x
    @params: {float} y
    return: np.exp(- ((x**2 + y**2) / (2 * sigma**2))) / (2 * np.pi * sigma**2)
    '''
    return np.exp(- ((x**2 + y**2) / (2 * sigma**2))) / (2 * np.pi * sigma**2)

# def Gaussian_Derive_x(sigma, x, y):
#     return Gaussian_function(sigma, x, y) * (-x / sigma**2)

# def Gaussian_Derive_y(sigma, x, y):
#     return Gaussian_function(sigma, x, y) * (-y / sigma**2)

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


def Non_maximum(img, dir):
    '''
    description: make the rough edges become thin edges which has only one pixel in gradient direction
    @param {array[[]]} img
    @param {array[[]]} dir
    return {array[[]]} result_img
    '''
    new_img = np.zeros((img.shape[0] + 2, img.shape[1] + 2)) 
    new_img[1:img.shape[0]+1, 1:img.shape[1]+1] = np.copy(img) #strade the orinal image with zeros in order to index error
    result_img = np.copy(img) #initialize result image
    
    #Compare the value of center pixel and the neiboring pixels in the gradient direction. 
    #If the value of center pixel is less than its neighbors',
    #then it will be set 0. Otherwise, it can be preserved 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            theta = dir[i, j]
            pi = np.pi
            Gp1 = 0
            Gp2 = 0
            if(0 <= theta <= (pi//4) or (-pi) <= theta <= (-3*pi // 4)):
                if(theta < 0):
                    theta += pi
                Gp1 = (1-np.tan(theta)) * new_img[i+1, j+1 +1] + np.tan(theta) * new_img[i+1 -1,j+1 +1]
                Gp2 = (1-np.tan(theta)) * new_img[i+1, j+1 -1] + np.tan(theta) * new_img[i+1 +1,j+1 -1]
                
            elif((pi//4) <= theta <= (pi//2) or (-3*pi//4) <= theta <= (-pi//2)):
                theta = pi//2 - theta if theta > 0 else -pi//2 - theta
                Gp1 = (1-np.tan(theta)) * new_img[i+1 -1, j+1] + np.tan(theta) * new_img[i+1 -1,j+1 +1]
                Gp2 = (1-np.tan(theta)) * new_img[i+1 +1, j+1] + np.tan(theta) * new_img[i+1 +1,j+1 -1]

            elif((pi//2) <= theta <= (3*pi//4) or (-pi//2) <= theta <= (-pi//4)):
                theta = theta - pi//2 if theta > 0 else theta + pi//2
                Gp1 = (1-np.tan(theta)) * new_img[i+1 -1, j+1] + np.tan(theta) * new_img[i+1 -1,j+1 -1]
                Gp2 = (1-np.tan(theta)) * new_img[i+1 +1, j+1] + np.tan(theta) * new_img[i+1 +1,j+1 +1]
            
            else:
                theta = pi - theta if theta > 0 else - theta
                Gp1 = (1-np.tan(theta)) * new_img[i+1, j+1 -1] + np.tan(theta) * new_img[i+1 -1,j+1 -1]
                Gp2 = (1-np.tan(theta)) * new_img[i+1, j+1 +1] + np.tan(theta) * new_img[i+1 +1,j+1 +1]

            if(new_img[i+1, j+1] < Gp1 or new_img[i+1,j+1] < Gp2):
                result_img[i, j] = 0
    
    return result_img

def Hysteresis_thresholding(img, low_rate, high_rate):
    '''
    description: 
    @param {array[[]]} img
    @param {float} low_rate
    @param {float} high_rate
    return {array[[]]} flag_mat[1:img.shape[0]+1, 1:img.shape[1]+1]
    '''
    max_pixel = np.max(img)
    low, high = low_rate*max_pixel, high_rate*max_pixel
    flag_mat = np.zeros((img.shape[0]+2, img.shape[1]+2))
    new_img = np.zeros((img.shape[0]+2, img.shape[1]+2))
    new_img[1:img.shape[0]+1, 1:img.shape[1]+1] = np.copy(img)
    q1 = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if new_img[i+1,j+1] < low:
                flag_mat[i+1,j+1] = 0
            elif new_img[i+1,j+1] > high:
                flag_mat[i+1,j+1] = 255
            else:
                q1.append((i+1,j+1))
                flag_mat[i+1,j+1] = -1

    
    while(True):
        q2 = []
        flag = 0 #用于记录是否有对矩阵更改
        while(len(q1) != 0):
            (i,j) = q1.pop()
            if(flag_mat[i-1,j-1] or flag_mat[i-1,j] or flag_mat[i-1,j+1] or flag_mat[i,j-1] or flag_mat[i,j+1] or flag_mat[i+1,j-1] or flag_mat[i+1,j] or flag_mat[i+1,j+1]):
                flag_mat[i,j] = 255
                flag = 1
            elif(flag_mat[i-1,j-1] == 0 and flag_mat[i-1,j] == 0 and flag_mat[i-1,j+1] == 0 and flag_mat[i,j-1] == 0 and flag_mat[i,j+1] == 0 and flag_mat[i+1,j-1] == 0 and flag_mat[i+1,j] == 0 and flag_mat[i+1,j+1] == 0):
                flag_mat[i,j] = 0
            else:
                q2.append((i,j))
        
        if(len(q2) != 0 or flag == 1):
            q1 = q2[:]
        else:
            break
    
    flag_mat = np.array(flag_mat)
    flag_mat[flag_mat==-1] = 0

    ic(flag_mat[1:img.shape[0]+1, 1:img.shape[1]+1].shape)
    return flag_mat[1:img.shape[0]+1, 1:img.shape[1]+1]


def Gaussian_Smooth(img, sigma, kernelsize):
    Gaussian_Kernel = Get_Gaussian_Kernel(sigma, kernelsize)
    new_img = conv(img, Gaussian_Kernel)
    new_img = np.rint(new_img.clip(0, 255)).astype('uint8')
    return new_img


def Sobel(img, sigma, kernelsize):
    '''
    description: Extract the edge of the picture
    @params:{np.array[[]]} a raw img
    return:{np.array[[]]} a new img after Gaussion filtering
    '''
    Gaussian_Smooth_img = Gaussian_Smooth(img, sigma, kernelsize)

    Gx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    Gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    img_x = conv(Gaussian_Smooth_img, Gx)
    img_y = conv(Gaussian_Smooth_img, Gy)
    
    new_img = np.sqrt(img_x**2 + img_y**2)
    new_img.clip(0, 255)
    new_img = np.rint(new_img).astype('uint8')

    img_x = np.rint(img_x.clip(0, 255)).astype('uint8')
    img_y = np.rint(img_y.clip(0, 255)).astype('uint8')

 

    return img_x, img_y, new_img

def Prewitt(img, sigma, kernelsize):
    '''
    description: Extract the edge of the picture
    @params:{np.array[[]]} a raw img
    return:{np.array[[]]} a new img after Gaussion filtering
    '''
    Gaussian_Smooth_img = Gaussian_Smooth(img, sigma, kernelsize)

    Gx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])
    Gy = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]])
    img_x = conv(Gaussian_Smooth_img, Gx)
    img_y = conv(Gaussian_Smooth_img, Gy)
    
    new_img = np.sqrt(img_x**2 + img_y**2)
    new_img.clip(0, 255)
    new_img = np.rint(new_img).astype('uint8')

    img_x = np.rint(img_x.clip(0, 255)).astype('uint8')
    img_y = np.rint(img_y.clip(0, 255)).astype('uint8')


    return img_x, img_y, new_img

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
    plt.savefig('./Result/Canny_Detect' + result_path)
    plt.show()

    #Sobel Detect
    _, __, Sobel_img = Sobel(img, sigma, kernel_size)

    #Prewitt Detect
    _, __, Prewitt_img = Prewitt(img, sigma, kernel_size)

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


```

