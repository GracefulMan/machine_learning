import skimage.transform as st
import matplotlib.pyplot as plt
from skimage import data,feature,io
import numpy as np
from matplotlib.pyplot import MultipleLocator
#使用Probabilistic Hough Transform.
image = io.imread('22.png',as_gray=True)
H,W = image.shape[0],image.shape[1]
print(W,H)
#image = image[200:600,150:]
edges = feature.canny(image, sigma=1.1, low_threshold=0, high_threshold=1)
h, theta, d = st.hough_line(edges)
lines = st.probabilistic_hough_line(edges, threshold=5, line_length=50,line_gap=10)
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

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 6))
plt.tight_layout()
#显示原图像
ax0.imshow(image, plt.cm.gray)
ax0.set_title('Input image')
ax0.set_axis_off()
#显示canny边缘
ax1.imshow(edges, plt.cm.gray)
ax1.set_title('Canny edges')
ax1.set_axis_off()
#用plot绘制出所有的直线
temp_line = []
ax2.imshow(edges * 0)
for line in lines:
    p0, p1 = line
    k = (p1[0] - p0[0]) / (p1[1]-p0[1] + 0.00001)
    # 从直线中挑选出水平和数值的，对他们进行一次分类
    if np.abs(k) < 0.5 or np.abs(k) > 100:
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


# 这个函数其实没有必要,只是整理点用的
def get_points_from_line_set(lines):
    vertical_line = []
    horizontal_line = []
    for line in lines:
        if line[0] < 0.5:
            for item in line[1:]:
                vertical_line.append(item)
        else:
            for item in line[1:]:
                horizontal_line.append(item)
    return vertical_line,horizontal_line


vertical_line_points, horizontal_line_points = get_points_from_line_set(temp_line)


# 搜索对应阈值下的直线，将点按照直线进行分类
def search_parameters(points, threshold, mode = 'v'):
    # 确定最低点坐标和最高点坐标
    min_value, max_value = 1e6, -1e6
    parameters = []
    indicator = 1
    if mode == 'v': # vertical points
        indicator = 0

    # 寻找上下阈值
    for point in points:
        tmp_min_value = min(point[0][indicator],point[1][indicator])
        if min_value > tmp_min_value:
            min_value = tmp_min_value
        tmp_max_value = max(point[0][indicator],point[1][indicator])
        if max_value < tmp_max_value:
            max_value = tmp_max_value

    # 求解参数
    for parameter in range(min_index, max_value, threshold):
        s_n = 0
        point_num = 0
        for point in points:
            for i in range(2):
                if np.abs(point[i][indicator] - parameter) < max(threshold / 3, 5):
                    point_num += 1
                    s_n += (parameter - point[i][indicator])**2
        credibility = max(3.0/64 * threshold**2 * point_num, 10)
        #print(credibility,point_num)
        if s_n <= credibility and s_n > 0 and point_num > 2 :
            parameters.append(parameter)
    return parameters
tmp = search_parameters(vertical_line_points, 1)
print(tmp)
tmp2 = search_parameters(horizontal_line_points, 1, mode='h')
print(tmp2)

for b in tmp:
    print(b)
    ax2.plot([b, b], [0, H-1],color='r')
for b in tmp2:
    ax2.plot([0,W-1],[b,b],color='r')
row2, col2 = image.shape
ax2.axis((0, col2, row2, 0))
ax2.set_title('Probabilistic Hough line')
ax2.set_axis_off()
ax2.imshow(image)
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