from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from matplotlib.patches import Rectangle
from scipy.stats import linregress
import matplotlib.patches as patches
from matplotlib.lines import Line2D

frames = glob.glob("frames/frame_*.png")
frames.sort()  # This works because frame numbers are zero-padded
frames = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in frames]

fig,axes = plt.subplots(11,7,figsize=(28,44))
axes = axes.flatten()  # turn 2D array into 1D list of axes

def calculate_regression(points):
    # convert points to float
    points = points.astype(float)
    xs = points[:, 1]  # x coordinates
    ys = points[:, 0]  # y coordinates

    # Vertical line case
    x_var = np.var(xs)
    y_var = np.var(ys)

    if x_var < 1e-2 * y_var:
        x = np.mean(xs)
        return ('vertical', x)

    # Regular linear regression
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)
    xy_mean = np.mean(xs * ys)
    x_squared_mean = np.mean(xs ** 2)

    m = (x_mean * y_mean - xy_mean) / (x_mean ** 2 - x_squared_mean)
    b = y_mean - m * x_mean

    return (m, b)

def find_inliers(m_or_type, b_or_x, shape):
    x1, y1, x2, y2 = None, None, None, None # TODO
    height, width = shape

    if m_or_type == 'vertical':
        x = b_or_x
        if 0 <= x <= width:
            return x, 0, x, height
        else:
            return None

    m, b = m_or_type, b_or_x
    points = []

    y_left = b
    if 0 <= y_left <= height:
        points.append((0, y_left))
    y_right = m * width + b
    if 0 <= y_right <= height:
        points.append((width, y_right))
    x_top = -b / m if m != 0 else None
    if x_top is not None and 0 <= x_top <= width:
        points.append((x_top, 0))
    x_bottom = (height - b) / m if m != 0 else None
    if x_bottom is not None and 0 <= x_bottom <= width:
        points.append((x_bottom, height))

    if len(points) < 2:
        return None

    (x1,y1), (x2,y2) = points[:2]
    
    return x1, y1, x2, y2


for i, image in enumerate(frames):
    #dilate
    imgray = image.copy()
    kernel = np.ones((5,5),np.uint8);
    imgray = cv2.dilate(imgray,kernel,iterations = 1)

    #binary threshold and find contours
    ret, thresh = cv2.threshold(imgray, 253, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_box = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

    mask = np.zeros_like(thresh)
    min_ratio = 3.0  # Minimum aspect ratio to keep

    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(image_with_box,[box],0,(0,0,255),2)

        #filter out thresholded parts that have insufficient aspect ratio
        w,h = rect[1]
        if w == 0 or h == 0:
            continue
        aspect_ratio = max(w/h, h/w)
        if aspect_ratio > min_ratio:
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(mask,[box],0,255,-1)

    thresh = cv2.bitwise_and(thresh, mask)

    m,b = calculate_regression(np.argwhere(thresh))
    _ = find_inliers(m,b, thresh.shape)
    if _ is not None:
        regression = Line2D([_[0],_[2]],[_[1],_[3]], color='lime')
        ax.add_line(regression);
    
    ax = axes[i]
    ax.imshow(image_with_box, cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()