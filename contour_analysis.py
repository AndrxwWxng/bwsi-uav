
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from skimage.morphology import skeletonize
import os
import time


def contour_analysis(image): 
    kernel = np.ones((5,5), np.uint8)
    
    _, thresh = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, kernel, iterations=4)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    if not contours:
        print("No contours found")
        return None, None

    max_contour = contours[0]
    for c in contours:
        if max(cv2.boundingRect(c)[3], cv2.boundingRect(c)[2]) > max(cv2.boundingRect(max_contour)[3], cv2.boundingRect(max_contour)[2]):
            max_contour = c
    
    epsilon = 0.001 * cv2.arcLength(max_contour, True)
    max_contour = cv2.approxPolyDP(max_contour, epsilon, True)
    
   
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [max_contour], -1, 255, -1)

    masked_img = cv2.bitwise_and(image, mask)

    skeleton = skeletonize(masked_img)
    skeleton = skeleton.astype(np.uint8) * 255 
    skeleton = cv2.dilate(skeleton, kernel, iterations=5)
    skeleton = cv2.GaussianBlur(skeleton, (101, 101), 0)
    skeleton = cv2.threshold(skeleton, 100, 255, cv2.THRESH_BINARY)[1]
    skeleton = skeletonize(skeleton)
  
    points = np.column_stack(np.where(skeleton > 0))
    if points.shape[0] >= 100:
        n = points.shape[0]//100
    else:
        n = 2
    indices = np.linspace(0, len(points)-1, n, dtype=int)
    points = points[indices]
    
    return masked_img, points, max_contour



frames = [f for f in os.listdir("frames") if f.endswith(".png")]
for f in frames:
    start_time = time.time()

    image = cv2.imread(f"frames/{f}", cv2.IMREAD_GRAYSCALE)
    masked_img, points, max_contour = contour_analysis(image)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Processing time: {elapsed:.4f} seconds")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(masked_img, cmap='gray')
    axes[1].imshow(image, cmap='gray')
    axes[0].scatter(points[:, 1], points[:, 0], c='red')

    for i in range(len(points) - 1):
        y_values = [points[i, 0], points[i+1, 0]]
        x_values = [points[i, 1], points[i+1, 1]]
        axes[0].plot(x_values, y_values, color='blue')

    axes[0].set_aspect('equal')
    axes[0].set_xlim(0, image.shape[1]) 
    axes[0].set_ylim(image.shape[0],0) 
    plt.show()


    
