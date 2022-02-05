
import numpy as np

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