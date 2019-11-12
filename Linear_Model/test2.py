import skimage.transform as st
import matplotlib.pyplot as plt
from skimage import data,feature,io
import numpy as np
from matplotlib.pyplot import MultipleLocator
#使用Probabilistic Hough Transform.
image = io.imread('1.jpg',as_gray=True)
H,W = image.shape[0],image.shape[1]
print(W,H)
#image = image[200:600,150:]
edges = feature.canny(image, sigma=1.1, low_threshold=0, high_threshold=1)
h, theta, d = st.hough_line(edges)
lines = st.probabilistic_hough_line(edges, threshold=5, line_length=50,line_gap=5)
print(len(lines))
# 创建显示窗口.

def linear_regression(x, y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x ** 2)
    sumxy = sum(x * y)
    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
    return np.linalg.solve(A, b)

#交互
# fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 6))
# plt.tight_layout()
# #显示原图像
# ax0.imshow(image, plt.cm.gray)
# ax0.set_title('Input image')
# ax0.set_axis_off()
# #显示canny边缘
# ax1.imshow(edges, plt.cm.gray)
# ax1.set_title('Canny edges')
# ax1.set_axis_off()
# #用plot绘制出所有的直线
# ax2.imshow(edges * 0)

temp_line = []
for line in lines:
    p0, p1 = line
    k = (p1[0] - p0[0]) / (p1[1]-p0[1] + 0.00001)
    # 从直线中挑选出水平和数值的，对他们进行一次分类
    min_index = 0
    min_gap = 100000
    for i in range(len(temp_line)):
        if np.abs(temp_line[i][0] - k) < min_gap:
            min_gap = np.abs(temp_line[i][0] - k)
            min_index = i
    if min_gap > 1:
        temp_line.append([k,line])
    else:
        temp_line[min_index].append(line)
print(temp_line)
for line in lines:
    p0,p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1])) # x0,x1. y0,y1
plt.imshow(image,plt.cm.gray)
# row2, col2 = image.shape
# ax2.axis((0, col2, row2, 0))
pos=plt.ginput(4)

# select point
def get_line(lines, pos, threshold):
    k = [
        (pos[0][0] - pos[1][0])/(pos[0][1] - pos[1][1]),
        (pos[2][0] - pos[3][0]) / (pos[2][1] - pos[3][1])
        ]
    final_line = [[pos[0],pos[1]],[pos[2], pos[3]]]
    for line in lines:
        for i in range(2):
            if np.abs(k[i] - line[0]) < 1 :
                for point in line[1:]:
                    d = np.abs(point[0][0] - k[i] * point[0][1] + k[i] * pos[0][1] - pos[0][0])/np.sqrt(1+np.square(k[i]))
                    if d < threshold:
                        final_line[i].append(point[0])
                        final_line[i].append(point[1])
    return final_line[0],final_line[1]
first_line, second_line = get_line(temp_line,pos,2)

def extract_coor(points):
    x = []
    y = []
    for point in points:
        x.append(int(point[0]))
        y.append(int(point[1]))
    return np.array(x),np.array(y)

x1, y1 = extract_coor(first_line)
x2, y2 = extract_coor(second_line)
a0, a1 = linear_regression(x1,y1)
b0,b1 =  linear_regression(x2,y2)

_X=[150, 350]
_Y = [a0 + a1 * x for x in _X]

_X1 = [230,300]
_Y1 = [b0 + b1 * x for x in _X1]

for line in lines:
    p0,p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1])) # x0,x1. y0,y1
plt.plot(_X, _Y, 'ro', _X, _Y, 'b', linewidth=2)
plt.plot(_X1, _Y1, 'ro', _X1, _Y1, 'r', linewidth=2)
angle = np.round((180 / np.pi) * np.arctan(np.abs((a1-b1)/(1+a1*b1))),2)
plt.text(10,10,'theta='+str(angle))
plt.imshow(image,plt.cm.gray)
plt.show()







# ax2.set_title('Probabilistic Hough line')
# ax2.set_axis_off()
plt.show()
# x_major_locator=MultipleLocator(1)
# fig,(ax3) = plt.subplots(1,1,figsize=(8, 6))
# ax3.imshow(np.log(1 + h))
# ax3.xaxis.set_major_locator(x_major_locator)
# ax3.set_title('Hough transform')
# ax3.set_xlabel('Angles (degrees)')
# ax3.set_ylabel('Distance (pixels)')
# ax3.axis('image')
# plt.xlim([0,90])
# plt.tight_layout()
# plt.show()