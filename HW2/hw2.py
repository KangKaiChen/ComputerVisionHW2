import cv2
import numpy as np
import matplotlib.pyplot as plt

def gray(img):
# 獲取圖像尺寸
    height, width, channels = img.shape
    
    # 創建一個新的NumPy數組來保存灰度圖像
    gray_img = np.zeros((height, width), dtype=np.uint8)

    # 轉換為灰度圖像
    for i in range(height):
        for j in range(width):
            # 獲取像素值
            b, g, r = img[i, j]
            # 計算灰度值
            gray_value = 0.299 * r + 0.587 * g + 0.114 * b
            # 將灰度值保存到新的NumPy數組中
            gray_img[i,j] =int(gray_value)
            
    return gray_img
        
        

def mean_filter(image, kernel_size):
    # 計算卷積核的半徑
    radius = kernel_size // 2
    
    # 計算需要填充的邊界數量
    padding_size = radius
    
    # 創建一個新的NumPy數組來保存濾波後的圖像
    filtered_image = np.zeros_like(image)
    
    # 在圖像周圍添加填充
    padded_image = np.pad(image, padding_size, mode='constant')
    
    # 遍歷圖像像素
    for y in range(radius, padded_image.shape[0]-radius):
        for x in range(radius, padded_image.shape[1]-radius):
            # 計算區域內像素的平均值
            kernel_sum = 0
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    kernel_sum += padded_image[y+i, x+j]
            mean_value = kernel_sum // (kernel_size ** 2)
            
            # 將平均值設置為濾波後像素的值
            filtered_image[y-radius, x-radius] = mean_value
    
    # 返回濾波後的圖像
    return filtered_image

def median_filter(image, kernel_size):
    # 計算卷積核的半徑
    radius = kernel_size // 2
    
    # 創建一個新的NumPy數組來保存濾波後的圖像
    filtered_image = np.zeros_like(image)
    
    # 遍歷圖像像素
    for y in range(radius, image.shape[0]-radius):
        for x in range(radius, image.shape[1]-radius):
            # 獲取區域內像素的列表
            pixel_list = []
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    pixel_list.append(image[y+i, x+j])
            
            # 排序像素列表
            pixel_list = quick_sort(pixel_list)
            
            # 將中間的像素值設置為濾波後像素的值
            median_value = pixel_list[(kernel_size**2) // 2]
            filtered_image[y, x] = median_value
    
    # 返回濾波後的圖像
    return filtered_image

def Histogram(img):
    hist = np.zeros((256,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i,j]] += 1
    plt.bar(np.arange(256), hist, width=1)
    plt.title("Histogram - image")
    plt.savefig('noise_image_his.png')
    plt.show()
            

def Histogram1(img):
    hist = np.zeros((256,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i,j]] += 1
    plt.bar(np.arange(256), hist, width=1)
    plt.title("Histogram - image")
    plt.savefig('output1_his.png')
    plt.show()
    
def Histogram2(img):
    hist = np.zeros((256,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i,j]] += 1
    plt.bar(np.arange(256), hist, width=1)
    plt.title("Histogram - image")
    plt.savefig('output2_his.png')
    plt.show()
    
    
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)




if __name__ == '__main__':
    img = cv2.imread("test_img/noise_image.png")
    imgGray = gray(img)
   # cv2.imshow("output1",imgGray)
    img1=  mean_filter(imgGray,3)
    img2 = median_filter(imgGray,3)
    cv2.imshow("output1",img1)
    cv2.imshow("output2",img2)
    cv2.imwrite("output1.png",img1)
    cv2.imwrite("output2.png",img2)
    Histogram(img)
    Histogram1(img1)
    Histogram2(img2)
    cv2.waitKey(0)