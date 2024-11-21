import pywt
import numpy as np
import cv2
# import matplotlib.pyplot as plt

# blur.png
image = cv2.imread("fig.JPEG", 0)
image = np.array(image)
image = np.expand_dims(image, 0)
image = np.repeat(image, 2, 0)
print(image.shape)
print(111)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 进行二维离散小波变换（2D-DWT）
coeffs = pywt.dwt2(image, 'haar')

# 从结果中获取近似子带和细节子带
cA, (cH, cV, cD) = coeffs
print(cH.shape)


#小波变换之后，低频分量对应的图像
# cv2.imwrite('lena.png',np.uint8(cA/np.max(cA)*255))
# # 小波变换之后，水平方向上高频分量对应的图像
# cv2.imwrite('lena_h.png',np.uint8(cH/np.max(cH)*255))
# # 小波变换之后，垂直方向上高频分量对应的图像
# cv2.imwrite('lena_v.png',np.uint8(cV/np.max(cV)*255))
# # 小波变换之后，对角线方向上高频分量对应的图像
# cv2.imwrite('lena_d.png',np.uint8(cD/np.max(cD)*255))
# # 根据小波系数重构的图像
# rimg=pywt.idwt2((cA,(cH,cV,cD)),"haar")
# cv2.imwrite("rimg.png",np.uint8(rimg))

# 打印结果
print("近似子带：")
print(cA[0])
print(cA[1])
print("\n水平细节子带：")
print(cH)
print("\n垂直细节子带：")
print(cV)
print("\n对角细节子带：")
print(cD)

# 可视化结果
# plt.figure(figsize=(8, 8))
# plt.subplot(2, 2, 1)
# plt.imshow(cA, cmap='gray')
# plt.title('Approximation')
# plt.subplot(2, 2, 2)
# plt.imshow(cH, cmap='gray')
# plt.title('Horizontal Detail')
# plt.subplot(2, 2, 3)
# plt.imshow(cV, cmap='gray')
# plt.title('Vertical Detail')
# plt.subplot(2, 2, 4)
# plt.imshow(cD, cmap='gray')
# plt.title('Diagonal Detail')
# plt.savefig("fig.jpg")
# plt.show()

